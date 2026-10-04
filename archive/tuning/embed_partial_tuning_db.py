from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
from pathlib import Path


REQUIRED_PARAMS = (
    "q_beta",
    "h",
    "cusum_k",
    "max_cov_inflate",
    "cov_decay",
    "z_entry",
    "z_exit",
)

REQUIRED_METRICS = ("meanpl", "sharpe", "score", "ret")
MIN_TRAIN_SCORE = 2.0
MIN_VALIDATION_SCORE = 2.0
MIN_SCORE_RETENTION = 0.25


def _instrument_symbols(strategy_source: str) -> set[str]:
    tree = ast.parse(strategy_source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "I"
            for target in node.targets
        ):
            instruments = ast.literal_eval(node.value)
            if not isinstance(instruments, dict):
                break
            return {str(symbol) for symbol in instruments}
    raise ValueError("Could not find the strategy's instrument map I")


def _load_rows(
    database: Path,
    valid_symbols: set[str],
    filter_mode: str,
) -> tuple[list[tuple[object, ...]], int]:
    uri = f"{database.resolve().as_uri()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        if filter_mode == "train-only":
            # Deliberately do not select test_metrics in this mode.
            records = [
                (*record, None)
                for record in connection.execute(
                    """
                    SELECT
                        pair_name,
                        y_symbol,
                        x_symbol,
                        auto_params,
                        auto_metrics
                    FROM pair_results
                    WHERE auto_params IS NOT NULL
                    ORDER BY pair_name
                    """
                ).fetchall()
            ]
        else:
            records = connection.execute(
                """
                SELECT
                    pair_name,
                    y_symbol,
                    x_symbol,
                    auto_params,
                    auto_metrics,
                    test_metrics
                FROM pair_results
                WHERE auto_params IS NOT NULL
                ORDER BY pair_name
                """
            ).fetchall()

    seen: set[str] = set()
    rows: list[tuple[object, ...]] = []
    for (
        pair_name,
        y_symbol,
        x_symbol,
        raw_params,
        raw_train_metrics,
        raw_test_metrics,
    ) in records:
        if pair_name in seen:
            raise ValueError(f"Duplicate pair_name in database: {pair_name}")
        seen.add(pair_name)

        if y_symbol not in valid_symbols or x_symbol not in valid_symbols:
            raise ValueError(
                f"{pair_name} uses unknown symbols: {y_symbol}, {x_symbol}"
            )
        if y_symbol == x_symbol:
            raise ValueError(f"{pair_name} uses the same symbol for both legs")

        params = json.loads(raw_params)
        missing = [key for key in REQUIRED_PARAMS if key not in params]
        if missing:
            raise ValueError(f"{pair_name} is missing parameters: {missing}")

        train_metrics = json.loads(raw_train_metrics)
        missing_train = [
            key for key in REQUIRED_METRICS if key not in train_metrics
        ]
        if missing_train:
            raise ValueError(f"{pair_name} has incomplete train metrics: {missing_train}")

        train_score = float(train_metrics["score"])
        profitable_train = (
            float(train_metrics["meanpl"]) > 0.0
            and float(train_metrics["sharpe"]) > 0.0
            and float(train_metrics["ret"]) > 0.0
            and train_score >= MIN_TRAIN_SCORE
        )
        if filter_mode == "train-only":
            profitable_and_stable = profitable_train
        else:
            test_metrics = json.loads(raw_test_metrics)
            validation_metrics = test_metrics.get("auto")
            if not isinstance(validation_metrics, dict):
                raise ValueError(f"{pair_name} has no auto validation metrics")
            missing_validation = [
                key for key in REQUIRED_METRICS if key not in validation_metrics
            ]
            if missing_validation:
                raise ValueError(
                    f"{pair_name} has incomplete validation metrics: "
                    f"{missing_validation}"
                )
            validation_score = float(validation_metrics["score"])
            profitable_and_stable = (
                profitable_train
                and float(validation_metrics["meanpl"]) > 0.0
                and float(validation_metrics["sharpe"]) > 0.0
                and float(validation_metrics["ret"]) > 0.0
                and validation_score >= MIN_VALIDATION_SCORE
                and validation_score >= MIN_SCORE_RETENTION * train_score
            )
        if not profitable_and_stable:
            continue

        rows.append(
            (
                str(y_symbol),
                str(x_symbol),
                float(params["q_beta"]),
                float(params["cusum_k"]),
                float(params["h"]),
                float(params["max_cov_inflate"]),
                float(params["cov_decay"]),
                float(params["z_entry"]),
                float(params["z_exit"]),
                float(params.get("z_stop", 10.0)),
            )
        )

    if not rows:
        raise ValueError("No auto-tuned pairs passed the performance filter")
    return rows, len(records)


def _generated_section(
    rows: list[tuple[object, ...]], database: Path, filter_mode: str
) -> str:
    tuple_lines = "\n".join(
        "    ("
        f"{y!r}, {x!r}, {q_beta!r}, {cusum_k!r}, {h!r}, "
        f"{max_cov_inflate!r}, {cov_decay!r}, {z_entry!r}, "
        f"{z_exit!r}, {z_stop!r}"
        "),"
        for (
            y,
            x,
            q_beta,
            cusum_k,
            h,
            max_cov_inflate,
            cov_decay,
            z_entry,
            z_exit,
            z_stop,
        ) in rows
    )
    if filter_mode == "train-only":
        gate_comment = (
            f"# Gate: training score >= {MIN_TRAIN_SCORE:g}, with positive training\n"
            "# mean P&L, Sharpe, and return. Validation/test metrics were not read."
        )
    else:
        gate_comment = (
            f"# Gate: train score >= {MIN_TRAIN_SCORE:g}, "
            f"validation score >= {MIN_VALIDATION_SCORE:g},\n"
            f"# validation/train score retention >= {MIN_SCORE_RETENTION:.0%}, "
            "and positive\n"
            "# mean P&L, Sharpe, and return in both stored periods."
        )
    return f"""# BEGIN GENERATED AUTO PAIR PARAMETERS
# Source: {database.name}; performance-filtered rows: {len(rows)}
{gate_comment}
_TUNED_PAIR_PARAMS: tuple[
    tuple[str, str, float, float, float, float, float, float, float, float],
    ...,
] = (
{tuple_lines}
)
# END GENERATED AUTO PAIR PARAMETERS


def _build_configs() -> tuple[PairCfg, ...]:
    out: list[PairCfg] = []
    for idx, (
        y_symbol,
        x_symbol,
        q_beta,
        cusum_k,
        h,
        max_cov_inflate,
        cov_decay,
        z_entry,
        z_exit,
        z_stop,
    ) in enumerate(_TUNED_PAIR_PARAMS):
        out.append(
            PairCfg(
                idx=idx,
                y=I[y_symbol],
                x=I[x_symbol],
                filt=FiltCfg(
                    q_beta=q_beta,
                    cusum_k=cusum_k,
                    h=h,
                    max_cov_inflate=max_cov_inflate,
                    cov_decay=cov_decay,
                ),
                trade=TradeCfg(
                    z_entry=z_entry,
                    z_exit=z_exit,
                    z_stop=z_stop,
                ),
            )
        )
    return tuple(out)


PAIR_CFGS = _build_configs()"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed completed auto-tuned pair rows into testbedv2_tuned.py"
    )
    parser.add_argument("--database", type=Path, default=Path("partial_tuning_data.db"))
    parser.add_argument("--strategy", type=Path, default=Path("testbedv2_tuned.py"))
    parser.add_argument(
        "--filter-mode",
        choices=("train-validation", "train-only"),
        default="train-validation",
        help="Metrics allowed to influence pair inclusion",
    )
    args = parser.parse_args()

    source = args.strategy.read_text(encoding="utf-8")
    valid_symbols = _instrument_symbols(source)
    rows, completed_count = _load_rows(
        args.database, valid_symbols, args.filter_mode
    )
    replacement = _generated_section(rows, args.database, args.filter_mode)

    pattern = re.compile(
        r"(?ms)^(?:# BEGIN GENERATED AUTO PAIR PARAMETERS.*?"
        r"# END GENERATED AUTO PAIR PARAMETERS\s+)?"
        r"def _build_configs\(\) -> tuple\[PairCfg, \.\.\.\]:.*?"
        r"^PAIR_CFGS = _build_configs\(\)"
    )
    updated, replacements = pattern.subn(replacement, source, count=1)
    if replacements != 1:
        raise ValueError("Could not locate exactly one pair-configuration section")

    temp_path = args.strategy.with_suffix(args.strategy.suffix + ".tmp")
    temp_path.write_text(updated, encoding="utf-8", newline="\n")
    temp_path.replace(args.strategy)
    print(
        f"Embedded {len(rows)} of {completed_count} auto-tuned pairs "
        f"into {args.strategy}"
    )


if __name__ == "__main__":
    main()
