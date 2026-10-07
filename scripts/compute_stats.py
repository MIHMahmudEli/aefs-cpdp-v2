import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

# Load data
aeeem = pd.read_csv(RESULTS / "aeeem_stage4_full_results.csv")
tera = pd.read_csv(RESULTS / "tera_stage4_full_results.csv")
try:
    nasa = pd.read_csv(RESULTS / "nasa_stage4_full_results.csv")
    has_nasa = True
except FileNotFoundError:
    print("NOTE: nasa_stage4_full_results.csv not found — skipping NASA section")
    has_nasa = False

try:
    relink = pd.read_csv(RESULTS / "relink_stage4_full_results.csv")
    has_relink = True
except FileNotFoundError:
    print("NOTE: relink_stage4_full_results.csv not found — skipping ReLink section")
    has_relink = False

datasets = [("AEEEM", aeeem), ("TeraPromise", tera)]
if has_nasa:
    datasets.append(("NASA MDP", nasa))
if has_relink:
    datasets.append(("ReLink", relink))

methods = ['AEFS', 'Ablation_no2D', 'AllFeatures', 'FilterMI', 'WrapperRFE']
models = ['lightgbm', 'random_forest']

# ============================================================
# SUMMARY TABLES
# ============================================================
for ds_name, df in datasets:
    print("=" * 80)
    print(f"{ds_name} RESULTS")
    print("=" * 80)
    for model in models:
        print(f"\n--- {model.upper()} ---")
        for method in methods:
            subset = df[(df['method'] == method) & (df['model'] == model)]
            mcc_mean = subset['test_mcc'].mean()
            mcc_std = subset['test_mcc'].std()
            f1_mean = subset['test_f1'].mean()
            auc_mean = subset['test_auc_roc'].mean()
            n_feat = subset['n_features'].mean()
            jaccard = subset['mean_jaccard'].dropna().mean() if subset['mean_jaccard'].notna().any() else float('nan')
            print(f"  {method:20s} | n_feat={n_feat:5.1f} | J={jaccard:.3f} | MCC={mcc_mean:.3f}\u00b1{mcc_std:.3f} | F1={f1_mean:.3f} | AUC={auc_mean:.3f}")

# ============================================================
# WILCOXON SIGNED-RANK TESTS
# ============================================================
print("\n" + "=" * 80)
print("WILCOXON SIGNED-RANK TESTS (AEFS vs each baseline)")
print("=" * 80)

for ds_name, df in datasets:
    print(f"\n--- {ds_name} ---")
    for model in models:
        print(f"\n  {model.upper()}:")
        aefs_vals = df[(df['method'] == 'AEFS') & (df['model'] == model)]['test_mcc'].values
        for baseline in ['Ablation_no2D', 'AllFeatures', 'FilterMI', 'WrapperRFE']:
            baseline_vals = df[(df['method'] == baseline) & (df['model'] == model)]['test_mcc'].values
            if len(aefs_vals) == len(baseline_vals) and len(aefs_vals) > 1:
                try:
                    stat, pval = stats.wilcoxon(aefs_vals, baseline_vals, alternative='two-sided')
                    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                    # Cliff's delta
                    n1, n2 = len(aefs_vals), len(baseline_vals)
                    delta = (sum(1 for a in aefs_vals for b in baseline_vals if a > b) -
                             sum(1 for a in aefs_vals for b in baseline_vals if a < b)) / (n1 * n2)
                    eff = 'negl.' if abs(delta) < 0.147 else 'small' if abs(delta) < 0.33 else 'medium' if abs(delta) < 0.474 else 'large'
                    print(f"    vs {baseline:20s}: W={stat:.1f}, p={pval:.4f} {sig}, d={delta:+.3f} ({eff})")
                except ValueError as e:
                    print(f"    vs {baseline:20s}: {e}")
            else:
                print(f"    vs {baseline:20s}: n={len(aefs_vals)} (too few)")

# ============================================================
# FRIEDMAN TEST
# ============================================================
print("\n" + "=" * 80)
print("FRIEDMAN TEST (omnibus across all 5 methods)")
print("=" * 80)

for ds_name, df in datasets:
    print(f"\n--- {ds_name} ---")
    for model in models:
        pivot = df[df['model'] == model].pivot_table(
            index='target', columns='method', values='test_mcc', aggfunc='first'
        ).dropna()
        method_cols = [m for m in methods if m in pivot.columns]
        data_matrix = pivot[method_cols].values

        if data_matrix.shape[0] > 2:
            stat, pval = stats.friedmanchisquare(*[data_matrix[:, i] for i in range(data_matrix.shape[1])])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            print(f"  {model.upper():12s}: chi2={stat:.2f}, p={pval:.4f} {sig}")
        else:
            print(f"  {model.upper():12s}: n={data_matrix.shape[0]} (too few)")

# ============================================================
# CROSS-DATASET SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("CROSS-DATASET AEFS SUMMARY (LightGBM)")
print("=" * 80)
print(f"  {'Dataset':15s} | {'Projects':>8s} | {'MCC':>8s} | {'F1':>8s} | {'AUC':>8s} | {'n_feat':>6s} | {'J':>6s}")
print("  " + "-" * 75)
for ds_name, df in datasets:
    s = df[(df['method'] == 'AEFS') & (df['model'] == 'lightgbm')]
    n_proj = len(s)
    j = s['mean_jaccard'].dropna().mean()
    jstr = f"{j:.3f}" if not np.isnan(j) else "  ---"
    print(f"  {ds_name:15s} | {n_proj:8d} | {s['test_mcc'].mean():8.3f} | {s['test_f1'].mean():8.3f} | {s['test_auc_roc'].mean():8.3f} | {s['n_features'].mean():6.1f} | {jstr}")

# ============================================================
# HEATMAP DATA (TeraPromise, AEFS, LightGBM)
# ============================================================
print("\n" + "=" * 80)
print("HEATMAP DATA (TeraPromise, AEFS, LightGBM)")
print("=" * 80)
heatmap = tera[(tera['method'] == 'AEFS') & (tera['model'] == 'lightgbm')][['target', 'test_mcc']].sort_values('target')
for _, row in heatmap.iterrows():
    print(f"  {row['target']:25s} MCC={row['test_mcc']:.3f}")
