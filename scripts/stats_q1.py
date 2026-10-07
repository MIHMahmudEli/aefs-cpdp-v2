"""Q1 statistics upgrade for the AEFS paper.

Adds, on top of scripts/compute_stats.py:
  1. Holm-Bonferroni corrected Wilcoxon p-values (within each pool x model family)
  2. Nemenyi post-hoc following a significant Friedman test (critical difference)
  3. Power justification: minimum detectable paired effect size (dz) for the
     realised pool sizes, plus achieved power for the observed effects

Run:  python scripts/stats_q1.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
POOLS = [("AEEEM", "aeeem"), ("TeraPromise", "tera"),
         ("NASA MDP", "nasa"), ("ReLink", "relink")]
BASELINES = ["Ablation_no2D", "AllFeatures", "FilterMI", "WrapperRFE"]
METHODS = ["AEFS", "Ablation_no2D", "AllFeatures", "FilterMI", "WrapperRFE"]
MODELS = ["lightgbm", "random_forest"]
ALPHA = 0.05
POWER_TARGET = 0.80


def load():
    out = []
    for name, stem in POOLS:
        p = RESULTS / f"{stem}_stage4_full_results.csv"
        if p.exists():
            out.append((name, pd.read_csv(p)))
    return out


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni step-down; returns adjusted p-values keyed as input."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, (m - i) * p)
        adj[k] = min(1.0, running)
    return adj


def cliffs_delta(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return (gt - lt) / (len(a) * len(b))


def min_detectable_dz(n: int, alpha: float = ALPHA, power: float = POWER_TARGET) -> float:
    """Smallest standardised paired effect detectable with `power` at `n` pairs."""
    df = n - 1
    if df < 1:
        return float("nan")
    tcrit = stats.t.ppf(1 - alpha / 2, df)

    def pow_at(dz):
        ncp = dz * np.sqrt(n)
        return (stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))

    lo, hi = 0.0, 10.0
    if pow_at(hi) < power:
        return float("nan")
    for _ in range(200):
        mid = (lo + hi) / 2
        if pow_at(mid) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def achieved_power(dz: float, n: int, alpha: float = ALPHA) -> float:
    df = n - 1
    if df < 1 or not np.isfinite(dz):
        return float("nan")
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    ncp = dz * np.sqrt(n)
    return stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp)


def sig_stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "$<$.001"
    return f"{p:.3f}"


def p_cell(p: float) -> str:
    core = fmt_p(p)
    if not np.isfinite(p):
        return core
    return core + ("$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01
                   else "$^{*}$" if p < 0.05 else "")


def d_cell(delta: float) -> str:
    return f"+{delta:.3f}" if delta >= 0 else f"$-${abs(delta):.3f}"


def latex_row(label: str, p_raw: float, p_holm: float, delta: float) -> str:
    label = label.replace("_", "\\_")
    eff = ("negl." if abs(delta) < 0.147 else "small" if abs(delta) < 0.33
           else "medium" if abs(delta) < 0.474 else "large")
    return (f"\\quad {label} & {p_cell(p_raw)} & {p_cell(p_holm)} & "
            f"{d_cell(delta)} & {eff} \\\\")


# ---------------------------------------------------------------- Wilcoxon
print("=" * 84)
print("1. WILCOXON + HOLM-BONFERRONI (family = pool x model, m = #baselines)")
print("=" * 84)
datasets = load()
latex = {}
for ds_name, df in datasets:
    print(f"\n--- {ds_name} ---")
    for model in MODELS:
        aefs = df[(df.method == "AEFS") & (df.model == model)]
        raw, hol, deltas, ws = {}, {}, {}, {}
        for bl in BASELINES:
            base = df[(df.method == bl) & (df.model == model)]
            if len(base) == 0 or len(base) != len(aefs) or len(aefs) < 3:
                continue
            a, b = aefs.test_mcc.values, base.test_mcc.values
            if np.allclose(a, b):
                raw[bl] = np.nan
                continue
            stat, p = stats.wilcoxon(a, b, alternative="two-sided")
            raw[bl] = p
            ws[bl] = stat
            deltas[bl] = cliffs_delta(a, b)
        finite = {k: v for k, v in raw.items() if np.isfinite(v)}
        hol = holm(finite)
        print(f"  {model.upper():12s} (family m={len(finite)})")
        for bl in BASELINES:
            if bl not in raw:
                continue
            p = raw[bl]
            if not np.isfinite(p):
                print(f"    vs {bl:16s}: all differences zero -> p undefined")
                continue
            d = deltas[bl]
            eff = ("negl." if abs(d) < 0.147 else "small" if abs(d) < 0.33
                   else "medium" if abs(d) < 0.474 else "large")
            print(f"    vs {bl:16s}: p={p:.4f}{sig_stars(p):>4s}  "
                  f"p_holm={hol[bl]:.4f}{sig_stars(hol[bl]):>4s}  "
                  f"d={d:+.3f} ({eff})")
        if model == "lightgbm":
            for bl in BASELINES:
                if bl in raw and np.isfinite(raw[bl]):
                    latex.setdefault(ds_name, []).append(
                        latex_row(bl, raw[bl], hol[bl], deltas[bl]))
                elif bl in raw:
                    print(f"    (latex: {bl} omitted, p undefined)")
                else:
                    print(f"    (latex: {bl} omitted, baseline not run)")

# ------------------------------------------------------------------ Friedman
print("\n" + "=" * 84)
print("2. FRIEDMAN + NEMENYI POST-HOC (critical difference)")
print("=" * 84)
q_alpha = stats.studentized_range.ppf(1 - ALPHA, 5, np.inf) / np.sqrt(2)
print(f"  q_alpha (k=5, alpha=0.05, Demsar convention) = {q_alpha:.4f}")

for ds_name, df in datasets:
    print(f"\n--- {ds_name} ---")
    for model in MODELS:
        pivot = (df[df.model == model]
                 .pivot_table(index="target", columns="method",
                              values="test_mcc", aggfunc="first").dropna())
        cols = [m for m in METHODS if m in pivot.columns]
        mat = pivot[cols].values
        if mat.shape[0] < 3:
            print(f"  {model.upper():12s}: n={mat.shape[0]} (too few)")
            continue
        stat, p = stats.friedmanchisquare(*[mat[:, i] for i in range(mat.shape[1])])
        N, k = mat.shape[0], mat.shape[1]
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * N))
        print(f"  {model.upper():12s}: chi2={stat:.2f}, p={p:.4f} "
              f"{sig_stars(p)}  N={N} k={k}  CD(0.05)={cd:.3f}")
        if p >= ALPHA:
            continue
        ranks = np.apply_along_axis(lambda r: stats.rankdata(-r), 1, mat)
        mean_rank = pd.Series(ranks.mean(axis=0), index=cols).sort_values()
        print("    mean ranks (1 = best): " + ", ".join(
            f"{m}={v:.3f}" for m, v in mean_rank.items()))
        aefs_rank = ranks[:, cols.index("AEFS")].mean() if "AEFS" in cols else np.nan
        for bl in cols:
            if bl == "AEFS":
                continue
            diff = abs(aefs_rank - ranks[:, cols.index(bl)].mean())
            flag = "EXCEEDS CD" if diff > cd else "within CD"
            print(f"    AEFS vs {bl:16s}: |rank diff|={diff:.3f}  "
                  f"CD={cd:.3f} -> {flag}")

# -------------------------------------------------------------------- power
print("\n" + "=" * 84)
print(f"3. POWER JUSTIFICATION (alpha=0.05 two-sided, target power={POWER_TARGET})")
print("=" * 84)
print("  minimum detectable standardised paired effect dz by pool size:")
for n in [3, 4, 5, 10, 20, 68]:
    print(f"    n={n:>2d}:  dz_min={min_detectable_dz(n):.3f}   "
          f"(equivalently: 95% CI half-width ~{1.96/np.sqrt(n):.2f} sd)")
print("\n  achieved power for the observed LightGBM pairwise effects:")
for ds_name, df in datasets:
    aefs = df[(df.method == "AEFS") & (df.model == "lightgbm")]
    n = len(aefs)
    for bl in BASELINES:
        base = df[(df.method == bl) & (df.model == "lightgbm")]
        if len(base) != n or n < 3:
            continue
        diff = aefs.test_mcc.values - base.test_mcc.values
        if np.allclose(diff, 0):
            continue
        dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.inf
        print(f"    {ds_name:13s} vs {bl:16s} n={n:>2d}  dz={dz:+.3f}  "
              f"power={achieved_power(dz, n):.3f}")

# -------------------------------------------------------------------- latex
print("\n" + "=" * 84)
print("4. LATEX ROWS FOR tab:wilcoxon (LightGBM, with Holm column)")
print("=" * 84)
for ds_name, _ in datasets:
    print(f"\n% {ds_name}")
    for row in latex.get(ds_name, []):
        print(row)
