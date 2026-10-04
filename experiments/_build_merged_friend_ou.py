#!/usr/bin/env python
"""Build merged_friend_ou.py from friend_book_latest + OU winners.

Skeleton: friend's latest portfolio layer (ridge-confirm, deadlock overlay 1.0,
ALGO residual, bang-bang voting).

Engines:
  - zband: friend's profiles for non-winner pairs
  - ou: tuned optimal-stopping for the 31 catalog OU winners, plus a small
    justified extra set from the screened OU cohorts (verified_new +
    ALUT-CCNS + RTTH-LSST) to bring the OU sleeve toward ~35.

Does not touch the live 80-pair OU search tuner.
"""
from __future__ import annotations

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PY = ROOT / "merged_friend_ou.py"
OUT_JSON = ROOT / "auto_tuning_results" / "merged_friend_ou_catalog.json"
OUT_META = ROOT / "auto_tuning_results" / "merged_friend_ou_meta.json"

# Hard set from _pair_attribution (tuned OU vs hybrid zband desired OOS).
OU_WINNERS_31 = frozenset(
    {
        "MTNS-BLBT",
        "CUBO-ULXY",
        "AMRP-DUCT",
        "NGTE-ILVX",
        "AETS-FCSG",
        "MDGI-ULXY",
        "ANSO-BENI",
        "HRET-HRND",
        "AENO-IHOZ",
        "RTTH-RRES",
        "AETS-ILVX",
        "SMAH-FWWG",
        "ALUT-HTRK",
        "ANSO-RCRI",
        "RTTH-SPLZ",
        "RTTH-RCRI",
        "HRET-MDGI",
        "MSDP-EAFC",
        "ANSO-RRES",
        "HRET-MTNS",
        "FWWG-NGTE",
        "ELLT-FCSG",
        "SMAH-ULXY",
        "RTTH-FWWG",
        "ANSO-RTTH",
        "ANSO-IHOZ",
        "CCNS-NAYO",
        "DUCT-HRND",
        "SRNA-NWIG",
        "LSST-EAFC",
        "GARI-FWWG",
    }
)

# Extra screened OU cohorts already in friend's book, individually strong OOS
# under OU (not in the 31 zband-vs-OU catalog comparison).
OU_EXTRA_JUSTIFIED = frozenset(
    {
        "ULXY-MSDP",  # verified_new / MSDP-ULXY orientation in friend catalog
        "MSDP-ULXY",
        "EELT-OTCS",
        "ALUT-CCNS",
        "RTTH-LSST",  # reject_cohort but +4.9k desired OOS under OU
    }
)

PARAM_KEYS = (
    "q_beta",
    "kalman_var_ewma_alpha",
    "cusum_k",
    "h",
    "max_cov_inflate",
    "cov_decay",
    "ou_window",
    "kappa_ewma_alpha",
    "sigma_ewma_alpha",
)


def ukey(y: str, x: str) -> frozenset[str]:
    return frozenset((y, x))


def name_of(y: str, x: str) -> str:
    return f"{y}-{x}"


def load_ou_param_index() -> dict[frozenset[str], dict]:
    """Unordered-pair -> tuned OU final_params (prefer catalog88, else merged)."""
    index: dict[frozenset[str], dict] = {}

    cat88 = ROOT / "auto_tuning_results" / "catalog88_tuning_results.json"
    if cat88.exists():
        for entry in json.loads(cat88.read_text(encoding="utf-8")):
            fp = entry.get("global", {}).get("final_params")
            if not fp:
                continue
            index[ukey(entry["y_symbol"], entry["x_symbol"])] = dict(fp)

    merged = ROOT / "auto_tuning_results" / "merged_catalog.json"
    if merged.exists():
        for row in json.loads(merged.read_text(encoding="utf-8")):
            p = row["params"]
            if p.get("engine") != "ou":
                continue
            k = ukey(row["y"], row["x"])
            # Prefer catalog88 when present; fill gaps from merged.
            if k not in index:
                index[k] = {key: p[key] for key in PARAM_KEYS}

    win = ROOT / "auto_tuning_results" / "windowed_tuning_results.json"
    if win.exists():
        for entry in json.loads(win.read_text(encoding="utf-8")):
            fp = entry.get("global", {}).get("final_params")
            if not fp:
                continue
            k = ukey(entry["y_symbol"], entry["x_symbol"])
            if k not in index:
                index[k] = dict(fp)

    return index


def build_catalog_rows() -> tuple[list[tuple], dict]:
    import friend_book_latest as fb

    ou_index = load_ou_param_index()
    ou_targets = set(OU_WINNERS_31) | set(OU_EXTRA_JUSTIFIED)

    rows: list[tuple] = []
    meta = {
        "ou_winners_31": sorted(OU_WINNERS_31),
        "ou_extra": [],
        "ou_applied": [],
        "zband_kept": [],
        "ou_missing_params": [],
    }

    for row in fb._ACTIVE_PAIR_PARAMS:
        y, x = row[0], row[1]
        n = name_of(y, x)
        rev = name_of(x, y)
        want_ou = n in ou_targets or rev in ou_targets
        k = ukey(y, x)
        fp = ou_index.get(k)

        if want_ou and fp is not None:
            vals = tuple(fp[key] for key in PARAM_KEYS)
            rows.append(("ou", y, x, *vals))
            tag = "winner31" if (n in OU_WINNERS_31 or rev in OU_WINNERS_31) else "extra"
            meta["ou_applied"].append({"pair": n, "tag": tag})
            if tag == "extra":
                meta["ou_extra"].append(n)
        else:
            if want_ou and fp is None:
                meta["ou_missing_params"].append(n)
            (
                q_beta,
                cusum_k,
                h,
                max_cov_inflate,
                cov_decay,
                z_entry,
                z_exit,
                z_stop,
            ) = row[2:]
            rows.append(
                (
                    "zband",
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
                )
            )
            meta["zband_kept"].append(n)

    meta["n_ou"] = sum(1 for r in rows if r[0] == "ou")
    meta["n_zband"] = sum(1 for r in rows if r[0] == "zband")
    meta["n_total"] = len(rows)
    return rows, meta


def emit_catalog_literal(rows: list[tuple]) -> str:
    lines = ["_MERGED_PAIR_PARAMS = ("]
    for row in rows:
        engine = row[0]
        y, x = row[1], row[2]
        vals = ", ".join(repr(v) for v in row[3:])
        lines.append(f'    ("{engine}", "{y}", "{x}", {vals}),')
    lines.append(")")
    return "\n".join(lines)


OU_HELPERS = r'''
TRANSACTION_COST = 0.0001
Z_ENTRY_RULE = "new"


class OUCfg(NamedTuple):
    ou_window: int = 100
    kappa_ewma_alpha: float = 0.1
    sigma_ewma_alpha: float = 0.1


def evaluate_ou_series(a: float, eff_c: float, N: int = 30) -> float:
    """Leung-Li V(a) residual used to solve optimal entry a*."""
    if a <= 0.0:
        return 1.0e6
    terms = []
    for n in range(N):
        log_num = math.lgamma(n + 0.5) - math.lgamma(0.5) - math.lgamma(n + 1)
        coeff = math.exp(log_num) * (math.sqrt(2.0) * a) ** (2 * n + 1) / (2 * n + 1)
        terms.append(coeff / math.factorial(n))
    series = sum(terms)
    return series - (a + eff_c)


def get_optimal_z_entry(
    kappa: float,
    sigma: float,
    roundtrip_cost: float,
    rule: str = "new",
) -> float:
    del rule  # 'new' and conventional share the same a* solver here
    if kappa <= 1.0e-12 or sigma <= 1.0e-12:
        return 1.0
    ou_std = sigma / math.sqrt(2.0 * kappa)
    eff_c = roundtrip_cost / max(ou_std, 1.0e-12)
    try:
        root = root_scalar(
            lambda a: evaluate_ou_series(a, eff_c),
            bracket=[1.0e-6, 8.0],
        )
        return float(root.root) if root.converged else 1.0
    except Exception:
        return 1.0
'''


def patch_source(src: str, catalog_literal: str) -> str:
    # Imports
    if "import os" not in src:
        src = src.replace(
            "from __future__ import annotations\n\nfrom math import sqrt\n",
            "from __future__ import annotations\n\nimport json\nimport math\nimport os\n"
            "from math import sqrt\n",
        )
    if "from scipy.optimize import root_scalar" not in src:
        src = src.replace(
            "import numpy as np\n",
            "import numpy as np\nfrom scipy.optimize import root_scalar\n",
        )

    # Docstring
    src = re.sub(
        r'^"""[\s\S]*?"""',
        '"""Friend latest portfolio layer + OU engine on individually improved pairs.\n\n'
        "Portfolio/conflict: friend_book_latest (ridge-confirm dual ridge, deadlock\n"
        "overlay 1.0, protected boost, ALGO residual, bang-bang).\n"
        "Engines: z-band for the friend's non-winner rows; OU optimal-stopping for\n"
        "the 31 catalog winners (+ justified screened extras). Ablation knobs via\n"
        "MERGED_* env vars (CONSENSUS_MARGIN, PROTECTED_BOOST, FLIP_TO_FLAT,\n"
        "MAG_WEIGHTED, CATALOG_JSON).\n"
        '"""',
        src,
        count=1,
    )

    # FiltCfg: keep ewma_lambda for zband; add kalman_var_ewma_alpha for OU rows
    src = src.replace(
        "class FiltCfg(NamedTuple):\n"
        "    q_beta: float\n"
        "    ewma_lambda: float = 0.97\n"
        "    cusum_k: float = 0.5\n"
        "    h: float = 8.0\n"
        "    vol_cut: bool = True\n"
        "    max_cov_inflate: float = 25.0\n"
        "    cov_decay: float = 0.95\n",
        "class FiltCfg(NamedTuple):\n"
        "    q_beta: float\n"
        "    ewma_lambda: float = 0.97\n"
        "    kalman_var_ewma_alpha: float | None = None\n"
        "    cusum_k: float = 0.5\n"
        "    h: float = 8.0\n"
        "    vol_cut: bool = True\n"
        "    max_cov_inflate: float = 25.0\n"
        "    cov_decay: float = 0.95\n",
    )

    # Variance update: honour kalman_var_ewma_alpha when set (OU tuned rows)
    src = src.replace(
        "    def _update_variances(self, innov: float, resid: float) -> float:\n"
        "        lam = self.cfg.ewma_lambda\n",
        "    def _update_variances(self, innov: float, resid: float) -> float:\n"
        "        if self.cfg.kalman_var_ewma_alpha is not None:\n"
        "            lam = 1.0 - float(self.cfg.kalman_var_ewma_alpha)\n"
        "        else:\n"
        "            lam = self.cfg.ewma_lambda\n",
    )

    # PairCfg: add ou + engine
    src = src.replace(
        "class PairCfg(NamedTuple):\n"
        "    idx: int\n"
        "    y: int\n"
        "    x: int\n"
        "    filt: FiltCfg\n"
        "    trade: TradeCfg\n",
        "class PairCfg(NamedTuple):\n"
        "    idx: int\n"
        "    y: int\n"
        "    x: int\n"
        "    filt: FiltCfg\n"
        "    trade: TradeCfg | None = None\n"
        "    ou: \"OUCfg\" = None  # type: ignore[assignment]\n"
        "    engine: str = \"zband\"\n",
    )

    # Insert OU helpers after TradeCfg / before PairCfg — after PairCfg actually after WARMUP block
    if "def get_optimal_z_entry" not in src:
        anchor = "class TradeCfg(NamedTuple):\n    z_entry: float\n    z_exit: float\n    z_stop: float\n"
        if anchor not in src:
            raise RuntimeError("TradeCfg anchor missing")
        src = src.replace(anchor, anchor + "\n" + OU_HELPERS + "\n", 1)

    # Default ou cfg on PairCfg instances — fix None default by post-init pattern in builders

    # Replace PairAgent with dual-engine version: patch catch_up trading block
    old_trade_block = '''            self.spread_hist.append(
                float(np.nan_to_num(output.spread, nan=0.0))
            )
            self._update_dynamic_ou()

            trade = self.spec.trade
            z = self.current_z
            abs_z = abs(z)

            if self.side == 0:
                if trade.z_entry <= abs_z < trade.z_stop:
                    self.side = (
                        int(-np.sign(z))
                        if z != 0.0
                        else 0
                    )
            else:
                exit_trade = False
                if self.side == 1 and z >= -trade.z_exit:
                    exit_trade = True
                elif self.side == -1 and z <= trade.z_exit:
                    exit_trade = True

                if exit_trade or abs_z >= trade.z_stop:
                    self.side = 0
'''
    new_trade_block = '''            self.spread_hist.append(
                float(np.nan_to_num(output.spread, nan=0.0))
            )

            if self.spec.engine == "zband":
                trade = self.spec.trade
                z = self.current_z
                abs_z = abs(z)
                if self.side == 0:
                    if trade.z_entry <= abs_z < trade.z_stop:
                        self.side = (
                            int(-np.sign(z))
                            if z != 0.0
                            else 0
                        )
                else:
                    exit_trade = False
                    if self.side == 1 and z >= -trade.z_exit:
                        exit_trade = True
                    elif self.side == -1 and z <= trade.z_exit:
                        exit_trade = True
                    if exit_trade or abs_z >= trade.z_stop:
                        self.side = 0
                continue

            self._update_dynamic_ou()

            if self.current_kappa > 0.0:
                ou_std = self.current_sigma / sqrt(2.0 * self.current_kappa)
                spread_z = (
                    (self.spread_hist[-1] - self.current_theta)
                    / max(ou_std, 1.0e-8)
                )
            else:
                spread_z = 0.0

            if self.side == 0:
                if spread_z >= self.a_star:
                    self.side = -1
                elif spread_z <= -self.a_star:
                    self.side = 1
            elif self.side == 1:
                if spread_z >= self.a_star:
                    self.side = -1
            elif self.side == -1:
                if spread_z <= -self.a_star:
                    self.side = 1
'''
    if old_trade_block not in src:
        raise RuntimeError("PairAgent trade block not found for patch")
    src = src.replace(old_trade_block, new_trade_block, 1)

    # PairAgent state: add current_theta, a_star to slots + reset
    src = src.replace(
        '        "current_kappa",\n'
        '        "current_sigma",\n'
        '        "current_cut",\n',
        '        "current_kappa",\n'
        '        "current_sigma",\n'
        '        "current_theta",\n'
        '        "a_star",\n'
        '        "current_cut",\n',
    )
    src = src.replace(
        "        self.current_kappa = 0.1\n"
        "        self.current_sigma = 0.02\n"
        "        self.current_cut = False\n",
        "        self.current_kappa = 0.1\n"
        "        self.current_sigma = 0.02\n"
        "        self.current_theta = 0.0\n"
        "        self.a_star = 1.0\n"
        "        self.current_cut = False\n",
    )

    # Replace _update_dynamic_ou body to use OUCfg + a*
    old_ou = '''    def _update_dynamic_ou(self) -> None:
        if len(self.spread_hist) < 30:
            return

        s = np.array(self.spread_hist[-_OU_WINDOW:])
        s_lag = s[:-1]
        s_t = s[1:]
        n = len(s_lag)
        mean_lag = np.mean(s_lag)
        mean_t = np.mean(s_t)
        denom = np.sum((s_lag - mean_lag) ** 2)

        if denom > 1.0e-8:
            b_mle = (
                np.sum(
                    (s_lag - mean_lag)
                    * (s_t - mean_t)
                )
                / denom
            )
            b_mle = min(max(b_mle, 1.0e-4), 0.9999)
            a_mle = mean_t - b_mle * mean_lag
            residuals = s_t - (a_mle + b_mle * s_lag)
            sigma_eps = np.sqrt(
                np.sum(residuals**2) / (n - 2)
            )

            raw_kappa = -np.log(b_mle)
            raw_sigma = sigma_eps * np.sqrt(
                (2.0 * raw_kappa)
                / (1.0 - np.exp(-2.0 * raw_kappa))
            )

            self.current_kappa = (
                0.9 * self.current_kappa
                + 0.1 * raw_kappa
            )
            self.current_sigma = (
                0.9 * self.current_sigma
                + 0.1 * raw_sigma
            )
'''
    new_ou = '''    def _update_dynamic_ou(self) -> None:
        ou = self.spec.ou if self.spec.ou is not None else OUCfg()
        min_samples = max(2, int(ou.ou_window))
        if len(self.spread_hist) < min_samples:
            return

        s = np.array(self.spread_hist[-min_samples:])
        s_lag = s[:-1]
        s_t = s[1:]
        n = len(s_lag)
        mean_lag = np.mean(s_lag)
        mean_t = np.mean(s_t)
        denom = np.sum((s_lag - mean_lag) ** 2)

        if denom > 1.0e-8:
            b_mle = (
                np.sum(
                    (s_lag - mean_lag)
                    * (s_t - mean_t)
                )
                / denom
            )
            b_mle = min(max(b_mle, 1.0e-4), 0.9999)
            a_mle = mean_t - b_mle * mean_lag
            residuals = s_t - (a_mle + b_mle * s_lag)
            sigma_eps = np.sqrt(
                np.sum(residuals**2) / (n - 2)
            )

            raw_kappa = -np.log(b_mle)
            raw_sigma = sigma_eps * np.sqrt(
                (2.0 * raw_kappa)
                / (1.0 - np.exp(-2.0 * raw_kappa))
            )
            theta = a_mle / (1.0 - b_mle)

            kappa_lambda = 1.0 - ou.kappa_ewma_alpha
            sigma_lambda = 1.0 - ou.sigma_ewma_alpha
            self.current_kappa = (
                kappa_lambda * self.current_kappa
                + (1.0 - kappa_lambda) * raw_kappa
            )
            self.current_sigma = (
                sigma_lambda * self.current_sigma
                + (1.0 - sigma_lambda) * raw_sigma
            )
            self.current_theta = (
                kappa_lambda * self.current_theta
                + (1.0 - kappa_lambda) * theta
            )
            self.a_star = get_optimal_z_entry(
                self.current_kappa,
                self.current_sigma,
                TRANSACTION_COST * 2.0,
                rule=Z_ENTRY_RULE,
            )
'''
    if old_ou not in src:
        raise RuntimeError("_update_dynamic_ou block not found")
    src = src.replace(old_ou, new_ou, 1)

    # Replace catalog section: from _SELECTED_PAIR_PARAMS through PAIR_CFGS =
    # Keep friend's merge/overlay logic by importing from friend_book_latest
    # at build time — embed engine-tagged rows instead.
    start = src.find("# The original union contained the complete requested")
    if start < 0:
        start = src.find("_SELECTED_PAIR_PARAMS = (")
    end = src.find("PAIR_CFGS = _build_selected_pair_cfgs()")
    if start < 0 or end < 0:
        raise RuntimeError("catalog section markers not found")
    end = end + len("PAIR_CFGS = _build_selected_pair_cfgs()")

    catalog_section = f'''
# Engine-tagged hybrid catalog (friend zband base + OU winners). Provenance:
# auto_tuning_results/merged_friend_ou_catalog.json
{catalog_literal}

if os.environ.get("MERGED_CATALOG_JSON"):
    with open(os.environ["MERGED_CATALOG_JSON"]) as _catalog_f:
        _MERGED_PAIR_PARAMS = tuple(
            tuple(_row) for _row in json.load(_catalog_f)
        )

_ACTIVE_PAIR_PARAMS = _MERGED_PAIR_PARAMS


def _build_selected_pair_cfgs() -> tuple[PairCfg, ...]:
    configs = []
    for idx, row in enumerate(_ACTIVE_PAIR_PARAMS):
        engine, y_symbol, x_symbol = row[0], row[1], row[2]
        if engine == "ou":
            (
                q_beta,
                kalman_var_ewma_alpha,
                cusum_k,
                h,
                max_cov_inflate,
                cov_decay,
                ou_window,
                kappa_ewma_alpha,
                sigma_ewma_alpha,
            ) = row[3:]
            configs.append(
                PairCfg(
                    idx=idx,
                    y=I[y_symbol],
                    x=I[x_symbol],
                    filt=FiltCfg(
                        q_beta=q_beta,
                        ewma_lambda=0.97,
                        kalman_var_ewma_alpha=kalman_var_ewma_alpha,
                        cusum_k=cusum_k,
                        h=h,
                        max_cov_inflate=max_cov_inflate,
                        cov_decay=cov_decay,
                    ),
                    trade=None,
                    ou=OUCfg(
                        ou_window=int(ou_window),
                        kappa_ewma_alpha=kappa_ewma_alpha,
                        sigma_ewma_alpha=sigma_ewma_alpha,
                    ),
                    engine="ou",
                )
            )
        elif engine == "zband":
            (
                q_beta,
                cusum_k,
                h,
                max_cov_inflate,
                cov_decay,
                z_entry,
                z_exit,
                z_stop,
            ) = row[3:]
            configs.append(
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
                    ou=OUCfg(),
                    engine="zband",
                )
            )
        else:
            raise ValueError(f"unknown engine tag {{engine!r}}")
    return tuple(configs)

PAIR_CFGS = _build_selected_pair_cfgs()

# Kept for the post-catalog integrity guards (pairs already filtered out of
# friend_book_latest._ACTIVE_PAIR_PARAMS before this hybrid catalog was built).
DEVELOPMENT_REMOVED_PAIR_NAMES = frozenset(
    {
        "ALGO-HRET",
        "SRNA-ELLT",
        "ULXY-EAFC",
    }
)
'''
    src = src[:start] + catalog_section + src[end:]

    # Conflict ablation knobs
    src = src.replace(
        "PROTECTED_VOTE_BOOST = 4.0\n"
        "RIDGE_CONFLICT_WEIGHT = 0.75\n"
        "USE_PAIR_CONFLICT_VOTES = True\n"
        "USE_RIDGE_CONFLICT_VOTES = True\n",
        "PROTECTED_VOTE_BOOST = float(os.environ.get(\"MERGED_PROTECTED_BOOST\", \"4.0\"))\n"
        "RIDGE_CONFLICT_WEIGHT = float(os.environ.get(\"MERGED_RIDGE_WEIGHT\", \"0.75\"))\n"
        "# Near-tie flat: suppress weak asset consensus instead of bang-bang flip.\n"
        "CONSENSUS_MARGIN = float(os.environ.get(\"MERGED_CONSENSUS_MARGIN\", \"0.15\"))\n"
        "# If >0, convert flipped legs to flat rather than opposite side.\n"
        "FLIP_TO_FLAT = os.environ.get(\"MERGED_FLIP_TO_FLAT\", \"0\") == \"1\"\n"
        "# Magnitude-weighted pair votes (else one vote per contributing pair).\n"
        "MAG_WEIGHTED_VOTES = os.environ.get(\"MERGED_MAG_WEIGHTED\", \"0\") == \"1\"\n"
        "# OU-specific conflict path (see merged_friend_ou sleeve_breakdown).\n"
        "# Rebuild note: structural OU conflict modes live in merged_friend_ou.py;\n"
        "# re-running this builder overwrites them — re-apply from git/history.\n"
        "OU_CONFLICT_MODE = os.environ.get(\"MERGED_OU_CONFLICT\", \"shared\").strip().lower()\n"
        "OU_SLEEVE_FRAC = float(os.environ.get(\"MERGED_OU_SLEEVE_FRAC\", \"0.5\"))\n"
        "OU_SLEEVE_SPLIT = os.environ.get(\"MERGED_OU_SLEEVE_SPLIT\", \"equal\").strip().lower()\n"
        "OU_SLEEVE_OPPOSE = os.environ.get(\"MERGED_OU_SLEEVE_OPPOSE\", \"net\").strip().lower()\n"
        "USE_PAIR_CONFLICT_VOTES = True\n"
        "USE_RIDGE_CONFLICT_VOTES = True\n",
    )

    # Magnitude-weighted votes in consensus
    src = src.replace(
        "        if PAIR_CONFLICT_MODE == \"priority\":\n"
        "            vote_weight = boost * max(float(priority), 1.0e-12)\n"
        "        else:\n"
        "            vote_weight = boost\n"
        "        for index, position in (\n"
        "            (int(agent.spec.y), pos_y),\n"
        "            (int(agent.spec.x), pos_x),\n"
        "        ):\n"
        "            if abs(position) <= 1.0e-9:\n"
        "                continue\n"
        "            numerator[index] += vote_weight * np.sign(position)\n"
        "            denominator[index] += vote_weight\n"
        "            contributor_count[index] += 1\n",
        "        if PAIR_CONFLICT_MODE == \"priority\":\n"
        "            vote_weight = boost * max(float(priority), 1.0e-12)\n"
        "        else:\n"
        "            vote_weight = boost\n"
        "        for index, position in (\n"
        "            (int(agent.spec.y), pos_y),\n"
        "            (int(agent.spec.x), pos_x),\n"
        "        ):\n"
        "            if abs(position) <= 1.0e-9:\n"
        "                continue\n"
        "            signed = np.sign(position)\n"
        "            w = vote_weight\n"
        "            if MAG_WEIGHTED_VOTES:\n"
        "                w = vote_weight * min(abs(position) / LEG, 1.0)\n"
        "            numerator[index] += w * signed\n"
        "            denominator[index] += w\n"
        "            contributor_count[index] += 1\n",
    )

    # After direction resolved, apply CONSENSUS_MARGIN and FLIP_TO_FLAT
    # Find the spot after _resolve_pair_ridge_direction in sleeve_breakdown
    marker = "    ) = _resolve_pair_ridge_direction(\n"
    idx = src.find(marker)
    if idx < 0:
        raise RuntimeError("resolve marker missing")
    # Find end of that call
    close = src.find("\n    )", idx)
    insert_at = close + len("\n    )")
    ablation = '''

    if CONSENSUS_MARGIN > 0.0:
        weak = (
            (np.abs(pair_vote) > 0.0)
            & (np.abs(pair_vote) < CONSENSUS_MARGIN)
            & pair_active
        )
        direction[weak] = 0.0

    if FLIP_TO_FLAT:
        pair_dir = np.sign(pair_vote)
        flipped = (
            pair_active
            & (pair_dir != 0.0)
            & (direction != 0.0)
            & (direction != pair_dir)
        )
        direction[flipped] = 0.0
'''
    src = src[:insert_at] + ablation + src[insert_at:]

    # Fix algo residual PairCfg to include engine/ou defaults
    src = src.replace(
        "                trade=TradeCfg(\n"
        "                    z_entry=z_entry,\n"
        "                    z_exit=z_exit,\n"
        "                    z_stop=z_stop,\n"
        "                ),\n"
        "            )\n"
        "        )\n"
        "    return tuple(configs)\n\n"
        "ALGO_RESIDUAL_CFGS",
        "                trade=TradeCfg(\n"
        "                    z_entry=z_entry,\n"
        "                    z_exit=z_exit,\n"
        "                    z_stop=z_stop,\n"
        "                ),\n"
        "                ou=OUCfg(),\n"
        "                engine=\"zband\",\n"
        "            )\n"
        "        )\n"
        "    return tuple(configs)\n\n"
        "ALGO_RESIDUAL_CFGS",
    )

    return src


def main() -> None:
    print(
        "WARNING: rebuilding merged_friend_ou.py from friend_book_latest will "
        "overwrite structural OU conflict modes (sleeve/ou_flat/twopass) that "
        "live only in merged_friend_ou.py — restore from git after catalog edits."
    )
    rows, meta = build_catalog_rows()
    OUT_JSON.write_text(json.dumps([list(r) for r in rows], indent=1), encoding="utf-8")
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"catalog: {meta['n_total']} pairs "
        f"(ou={meta['n_ou']}, zband={meta['n_zband']}, "
        f"extra={meta['ou_extra']}, missing={meta['ou_missing_params']})"
    )

    src = (ROOT / "friend_book_latest.py").read_text(encoding="utf-8")
    out = patch_source(src, emit_catalog_literal(rows))
    OUT_PY.write_text(out, encoding="utf-8")
    compile(out, str(OUT_PY), "exec")
    print(f"wrote {OUT_PY} ({out.count(chr(10))} lines)")


if __name__ == "__main__":
    main()
