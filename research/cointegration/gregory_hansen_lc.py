#!/usr/bin/env python
"""
Gregory-Hansen (1996) regime-shift cointegration + Hansen (1992) Lc stability.

Goal: explain what 'cointegration' is doing on these pairs —
stable link, break/regime shift, or unstable parameters (pseudo-coint).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

OUTPUT_DIR = Path("cointegration_output/gregory_hansen_lc")
TRIM = 0.15  # GH: search breaks in [trim, 1-trim]
SIGNIFICANCE = 0.05
MAX_LAGS = 4

# Asymptotic ADF* critical values, Gregory-Hansen (1996), m=1 regressor
# (reject no-coint if ADF* < critical; more negative = stronger)
GH_CRIT = {
    "C": {"1%": -5.13, "5%": -4.61, "10%": -4.34},       # level shift
    "C/T": {"1%": -5.45, "5%": -4.99, "10%": -4.72},     # level shift + trend
    "C/S": {"1%": -5.47, "5%": -4.95, "10%": -4.68},     # full regime shift
}

# Hansen (1992) Lc critical values (null: stable parameters). Reject stability if Lc > crit.
# Table for demeaned regression, k = number of I(1) regressors (excl. constant)
LC_CRIT = {
    1: {"10%": 0.353, "5%": 0.470, "1%": 0.748},
    2: {"10%": 0.600, "5%": 0.749, "1%": 1.074},
}

TOP_PAIRS = [
    ("ELLT", "ILVX"),
    ("AMRP", "HRET"),
    ("GARI", "IHOZ"),
    ("AENO", "RRES"),
    ("SRTX", "EAFC"),
    ("SMAH", "CTGI"),
    ("EORC", "NGTE"),
    ("HRET", "ULXY"),
    ("RTTH", "NGTE"),
    ("AENO", "NWIG"),
    ("GARI", "ULXY"),
    ("RRES", "ACAC"),
    ("ANSO", "ACIX"),
    ("CTGI", "HRND"),
    ("NWIG", "FWWG"),
    ("ALGO", "SMAH"),
    ("HRET", "ALUT"),
    ("SMAH", "HRND"),
]


def load_prices(path: str = "prices.txt") -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=0)


def _adf_t_on_resid(resid: np.ndarray, max_lags: int = MAX_LAGS) -> float:
    """ADF t-stat on residual (no constant — residuals already demeaned by construction)."""
    resid = np.asarray(resid, dtype=float)
    # Use statsmodels ADF with autolag; regression='n' (no constant) is correct for EG residuals
    try:
        stat, *_ = adfuller(resid, maxlag=max_lags, regression="n", autolag="AIC")
        return float(stat)
    except Exception:
        return np.nan


def _design_gh(y: np.ndarray, x: np.ndarray, break_idx: int, model: str) -> np.ndarray:
    """
    Regressors for Gregory-Hansen cointegrating regression at break (0-index, break starts at break_idx).
    Models:
      C   : const, D_t, x
      C/T : const, D_t, trend, x
      C/S : const, D_t, x, x*D_t
    """
    n = len(y)
    d = (np.arange(n) >= break_idx).astype(float)
    ones = np.ones(n)
    trend = np.arange(1, n + 1, dtype=float)
    if model == "C":
        return np.column_stack([ones, d, x])
    if model == "C/T":
        return np.column_stack([ones, d, trend, x])
    if model == "C/S":
        return np.column_stack([ones, d, x, x * d])
    raise ValueError(model)


def gregory_hansen(y: np.ndarray, x: np.ndarray, model: str = "C/S") -> dict:
    """
    Gregory-Hansen ADF* test: min ADF t-stat over candidate breaks.
    Null: no cointegration. Alt: cointegration with one break.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    lo = int(np.floor(n * TRIM))
    hi = int(np.ceil(n * (1 - TRIM)))
    best = {"adf_star": np.inf, "break_idx": None, "break_frac": None, "beta": None, "resid": None}

    for b in range(lo, hi + 1):
        Z = _design_gh(y, x, b, model)
        try:
            beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = y - Z @ beta
        adf_t = _adf_t_on_resid(resid)
        if np.isnan(adf_t):
            continue
        if adf_t < best["adf_star"]:
            best = {
                "adf_star": adf_t,
                "break_idx": b,
                "break_frac": b / n,
                "break_day": b,  # 0-based within series; caller maps to calendar
                "beta": beta,
                "resid": resid,
            }

    crit = GH_CRIT[model]
    adf_star = best["adf_star"]
    best.update(
        {
            "model": model,
            "n": n,
            "crit_5": crit["5%"],
            "crit_10": crit["10%"],
            "crit_1": crit["1%"],
            "reject_5": adf_star < crit["5%"],
            "reject_10": adf_star < crit["10%"],
            "reject_1": adf_star < crit["1%"],
        }
    )
    return best


def hansen_lc(y: np.ndarray, x: np.ndarray) -> dict:
    """
    Hansen (1992) Lc statistic for stability of cointegrating parameters.

    Null: constant cointegrating coefficients.
    Alt: parameters follow a martingale (unstable).
    Reject stability if Lc > critical value.

    Implementation: OLS cointegrating regression with constant;
    score f_t = z_t * e_t; Lc = tr( M^{-1} Omega^{-1} ... ) simplified bivariate form:
      S_t = sum_{i=1}^t f_i
      Lc = n^{-2} sum_t S_t' V^{-1} S_t
    with V = long-run variance of f_t (Newey-West).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    Z = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ beta

    # Scores: each row f_t = z_t * e_t  (k+1 dim including constant)
    f = Z * e[:, None]
    # Newey-West HAC of f (truncation ~ n^{1/3})
    L = max(1, int(np.floor(n ** (1 / 3))))
    Gamma0 = (f.T @ f) / n
    Omega = Gamma0.copy()
    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1.0)
        Gamma = (f[ell:].T @ f[:-ell]) / n
        Omega += w * (Gamma + Gamma.T)

    # Cumulative score
    S = np.cumsum(f, axis=0)
    try:
        Vinv = np.linalg.pinv(Omega)
    except np.linalg.LinAlgError:
        return {"Lc": np.nan, "reject_5": False, "crit_5": LC_CRIT[1]["5%"]}

    # Lc = n^{-2} sum_t S_t' Omega^{-1} S_t
    quad = np.einsum("ti,ij,tj->t", S, Vinv, S)
    Lc = float(np.sum(quad) / (n**2))

    k = 1  # one I(1) regressor
    crit = LC_CRIT[k]
    return {
        "Lc": Lc,
        "crit_10": crit["10%"],
        "crit_5": crit["5%"],
        "crit_1": crit["1%"],
        "reject_10": Lc > crit["10%"],
        "reject_5": Lc > crit["5%"],
        "reject_1": Lc > crit["1%"],
        "beta": beta,
        "resid": e,
    }


def classify(eg_sig: bool, gh_sig: bool, lc_unstable: bool) -> str:
    """
    Interpret the trio of tests.
    """
    if eg_sig and not lc_unstable:
        return "stable_coint"
    if (not eg_sig) and gh_sig:
        return "break_coint"  # only appears once a break is allowed
    if eg_sig and lc_unstable:
        return "unstable_coint"  # EG finds link but parameters wander
    if gh_sig and lc_unstable:
        return "break_and_unstable"
    if (not eg_sig) and (not gh_sig):
        return "no_coint"
    return "mixed"


def run_pair(prices: pd.DataFrame, a: str, b: str) -> dict:
    y = prices[a].values.astype(float)
    x = prices[b].values.astype(float)

    _, eg_p, _ = coint(y, x)
    eg_sig = eg_p < SIGNIFICANCE

    gh_results = {}
    for model in ("C", "C/T", "C/S"):
        gh_results[model] = gregory_hansen(y, x, model=model)

    # Prefer regime-shift model for headline; also keep best (most negative) ADF*
    best_model = min(gh_results.keys(), key=lambda m: gh_results[m]["adf_star"])
    gh_best = gh_results[best_model]

    lc = hansen_lc(y, x)
    # Also GH on regime model specifically
    gh_cs = gh_results["C/S"]

    label = classify(eg_sig, gh_best["reject_5"], lc["reject_5"])

    return {
        "pair": f"{a}-{b}",
        "ticker_a": a,
        "ticker_b": b,
        "eg_pval": eg_p,
        "eg_sig": eg_sig,
        "gh_best_model": best_model,
        "gh_adf_star": gh_best["adf_star"],
        "gh_crit_5": gh_best["crit_5"],
        "gh_sig": gh_best["reject_5"],
        "gh_break_frac": gh_best["break_frac"],
        "gh_break_day": int(gh_best["break_idx"]) + 1 if gh_best["break_idx"] is not None else None,
        "gh_C_sig": gh_results["C"]["reject_5"],
        "gh_CT_sig": gh_results["C/T"]["reject_5"],
        "gh_CS_sig": gh_results["C/S"]["reject_5"],
        "gh_CS_adf": gh_cs["adf_star"],
        "gh_CS_break_day": int(gh_cs["break_idx"]) + 1 if gh_cs["break_idx"] is not None else None,
        "lc": lc["Lc"],
        "lc_crit_5": lc["crit_5"],
        "lc_unstable": lc["reject_5"],
        "classification": label,
        "_gh_cs": gh_cs,
        "_lc": lc,
        "_y": y,
        "_x": x,
    }


def plot_pair_detail(res: dict, out_dir: Path) -> None:
    """Price levels + GH regime split + residual around break."""
    a, b = res["ticker_a"], res["ticker_b"]
    y, x = res["_y"], res["_x"]
    gh = res["_gh_cs"]
    br = gh["break_idx"]
    if br is None:
        return

    n = len(y)
    t = np.arange(1, n + 1)
    # Fitted under C/S
    Z = _design_gh(y, x, br, "C/S")
    yhat = Z @ gh["beta"]
    resid = gh["resid"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Normalized prices
    axes[0].plot(t, y / y[0], label=a, lw=1.2)
    axes[0].plot(t, x / x[0], label=b, lw=1.2)
    axes[0].axvline(br + 1, color="red", ls="--", lw=1.2, label=f"GH break day {br+1}")
    axes[0].set_ylabel("Price / day1")
    axes[0].set_title(
        f"{a}-{b} | class={res['classification']} | "
        f"EG p={res['eg_pval']:.3f} | GH ADF*={res['gh_CS_adf']:.2f} (crit {GH_CRIT['C/S']['5%']}) | "
        f"Lc={res['lc']:.3f} (crit {res['lc_crit_5']})"
    )
    axes[0].legend(fontsize=8, loc="best")

    axes[1].plot(t, y, color="#4C72B0", lw=1, label=a)
    axes[1].plot(t, yhat, color="#C44E52", lw=1, alpha=0.8, label="GH C/S fitted")
    axes[1].axvline(br + 1, color="red", ls="--", lw=1)
    axes[1].axvspan(1, br, color="#4C72B0", alpha=0.06)
    axes[1].axvspan(br + 1, n, color="#C44E52", alpha=0.06)
    axes[1].set_ylabel(a)
    axes[1].legend(fontsize=8)

    axes[2].plot(t, resid, color="#8172B2", lw=1)
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].axvline(br + 1, color="red", ls="--", lw=1)
    axes[2].set_ylabel("GH residual")
    axes[2].set_xlabel("Day")

    plt.tight_layout()
    path = out_dir / f"detail_{a}_{b}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_summary(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Classification counts
    order = ["stable_coint", "break_coint", "unstable_coint", "break_and_unstable", "no_coint", "mixed"]
    counts = df["classification"].value_counts().reindex(order, fill_value=0)
    colors = {
        "stable_coint": "#55A868",
        "break_coint": "#4C72B0",
        "unstable_coint": "#C44E52",
        "break_and_unstable": "#8172B2",
        "no_coint": "#999999",
        "mixed": "#CCB974",
    }
    axes[0].barh(counts.index, counts.values, color=[colors[k] for k in counts.index])
    axes[0].set_xlabel("# pairs")
    axes[0].set_title("What cointegration is doing (full sample)")

    # EG vs GH
    axes[1].scatter(
        df["eg_pval"].clip(upper=0.2),
        df["gh_CS_adf"],
        c=["#55A868" if s else "#C44E52" for s in df["lc_unstable"]],
        s=50,
        alpha=0.85,
    )
    axes[1].axvline(0.05, color="gray", ls="--", lw=1)
    axes[1].axhline(GH_CRIT["C/S"]["5%"], color="gray", ls="--", lw=1)
    axes[1].set_xlabel("Engle-Granger p-value (clipped at 0.2)")
    axes[1].set_ylabel("Gregory-Hansen ADF* (C/S)")
    axes[1].set_title("EG vs GH (green=Lc unstable)")
    for _, r in df.iterrows():
        axes[1].annotate(r["pair"], (min(r["eg_pval"], 0.2), r["gh_CS_adf"]), fontsize=6, alpha=0.7)

    # Lc
    axes[2].barh(df["pair"], df["lc"], color=["#C44E52" if u else "#55A868" for u in df["lc_unstable"]])
    axes[2].axvline(df["lc_crit_5"].iloc[0], color="black", ls="--", lw=1, label="5% Lc crit")
    axes[2].set_xlabel("Hansen Lc (higher = more unstable)")
    axes[2].set_title("Parameter stability (Hansen Lc)")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        "Gregory-Hansen + Hansen Lc — is cointegration stable, broken, or unstable?",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    path = out_dir / "gh_lc_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_break_timeline(df: pd.DataFrame, out_dir: Path) -> None:
    """Show estimated GH break days for pairs where GH finds coint."""
    sub = df[df["gh_CS_sig"] | df["gh_sig"]].copy()
    if sub.empty:
        return
    sub = sub.sort_values("gh_CS_break_day")
    fig, ax = plt.subplots(figsize=(11, max(4, 0.35 * len(sub))))
    ax.hlines(sub["pair"], 1, 750, color="#dddddd", lw=1)
    ax.scatter(sub["gh_CS_break_day"], sub["pair"], c="#C44E52", s=60, zorder=3)
    for _, r in sub.iterrows():
        ax.text(r["gh_CS_break_day"] + 5, r["pair"], f"d{int(r['gh_CS_break_day'])}", va="center", fontsize=7)
    ax.set_xlabel("Estimated Gregory-Hansen regime-shift day (C/S)")
    ax.set_xlim(0, 760)
    ax.set_title("Where the cointegrating regime breaks (full sample)")
    plt.tight_layout()
    path = out_dir / "gh_break_timeline.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    print(f"Loaded {len(prices)} days x {len(prices.columns)} instruments\n")

    rows = []
    details = []
    for a, b in TOP_PAIRS:
        res = run_pair(prices, a, b)
        print(
            f"{res['pair']:12s}  EG p={res['eg_pval']:.3f}({'Y' if res['eg_sig'] else 'n'})  "
            f"GH[{res['gh_best_model']}] ADF*={res['gh_adf_star']:.2f}"
            f"({'Y' if res['gh_sig'] else 'n'} @d{res['gh_break_day']})  "
            f"Lc={res['lc']:.3f}({'UNSTABLE' if res['lc_unstable'] else 'stable'})  "
            f"-> {res['classification']}"
        )
        plot_pair_detail(res, OUTPUT_DIR)
        details.append(res)
        rows.append({k: v for k, v in res.items() if not k.startswith("_")})

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "gh_lc_results.csv", index=False)
    plot_summary(df, OUTPUT_DIR)
    plot_break_timeline(df, OUTPUT_DIR)

    print("\n=== Classification counts ===")
    print(df["classification"].value_counts().to_string())
    print("\nLegend:")
    print("  stable_coint      : EG finds coint, Hansen Lc says parameters stable")
    print("  break_coint       : standard EG fails, but GH finds coint WITH a break")
    print("  unstable_coint    : EG finds coint, but Hansen Lc says coefficients wander")
    print("  break_and_unstable: GH finds regime shift AND Lc unstable")
    print("  no_coint          : neither EG nor GH finds cointegration")


if __name__ == "__main__":
    main()
