"""Compare the replicated runs against the local (authoritative) numbers.

    python scripts/cmp_replication.py

1. SHAP threshold sweep:  pinned Kaggle vs local, unpinned Kaggle vs local
   (68 targets x 4 thresholds = 272 cells each).
2. Stability threshold sweep: same.
3. Multi-seed run (seed 42): vs the local stage-4 results (68 targets).
4. Multi-seed seed spread:    reported in the manuscripts' sensitivity section.

Needs sensitivity/results{,_kaggle,_pinned,_multiseed}/ to be present.
"""

from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]
LOCAL = R / "sensitivity" / "results"
KG = R / "sensitivity" / "results_kaggle"
PIN = R / "sensitivity" / "results_pinned"
MSPIN = R / "sensitivity" / "results_multiseed"  # unpinned multi-seed
STAGE4 = R / "results" / "tera_stage4_full_results.csv"


def cells(a, b, label, key=("target", "shap_threshold")):
    a = pd.read_csv(a).sort_values(list(key)).reset_index(drop=True)
    b = pd.read_csv(b).sort_values(list(key)).reset_index(drop=True)
    key = list(key)
    if not a[key].equals(b[key]):
        print(f"  !! key mismatch ({label}): {len(a)} vs {len(b)} rows")
        return
    d = (a["mcc"] - b["mcc"]).abs()
    same_feat = int((a["n_features"] == b["n_features"]).sum())
    print(f"  {label:<34} rows={len(a):4d}  identical keys={len(a)}  "
          f"n_features identical={same_feat:3d}  "
          f"MCC within 0.05={(d <= 0.05).sum():3d}  "
          f"MCC >0.10={(d > 0.10).sum():3d}  "
          f"max|dMCC|={d.max():.4f}")


print("== SHAP threshold sweep (68 targets x 4 thresholds) ==")
cells(PIN / "tera_sensitivity_shap.csv", LOCAL / "tera_sensitivity_shap.csv",
      "pinned Kaggle vs local")
cells(KG / "tera_sensitivity_shap.csv", LOCAL / "tera_sensitivity_shap.csv",
      "unpinned Kaggle vs local")

print("== stability threshold sweep ==")
cells(PIN / "tera_sensitivity_stability.csv", LOCAL / "tera_sensitivity_stability.csv", "pinned Kaggle vs local", key=("target","stability_threshold"))
cells(KG / "tera_sensitivity_stability.csv", LOCAL / "tera_sensitivity_stability.csv", "unpinned Kaggle vs local", key=("target","stability_threshold"))


def multiseed(folder, label):
    fs = sorted(folder.glob("tera_multiseed_s*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    df = df[df["seed"] == 42].sort_values("target").reset_index(drop=True)
    s4 = pd.read_csv(STAGE4)
    s4 = s4[(s4["method"] == "AEFS") & (s4["model"] == "lightgbm")]
    s4 = s4[["target", "n_features", "test_mcc"]].rename(
        columns={"test_mcc": "mcc"}).sort_values("target").reset_index(drop=True)
    df = df[["target", "n_features", "mcc"]].sort_values(
        "target").reset_index(drop=True)
    if not df["target"].equals(s4["target"]):
        print(f"  !! target mismatch ({label}): {len(df)} vs {len(s4)}")
        return
    same_feat = int((df["n_features"] == s4["n_features"]).sum())
    d = (df["mcc"] - s4["mcc"]).abs()
    print(f"  {label:<34} rows={len(df):4d}  "
          f"n_features identical={same_feat:3d}  "
          f"MCC identical={(d < 5e-5).sum():3d}  "
          f"MCC within 0.05={(d <= 0.05).sum():3d}  "
          f"MCC >0.10={(d > 0.10).sum():3d}  "
          f"max|dMCC|={d.max():.4f}")


print("== multi-seed, seed 42 vs local stage-4 results ==")
multiseed(PIN, "pinned Kaggle vs local")
multiseed(MSPIN, "unpinned Kaggle vs local")

print("\n== pinned multi-seed spread (both pools) ==")
for pool in ("tera", "aeeem"):
    fs = sorted(PIN.glob(f"{pool}_multiseed_s*.csv"))
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    pm = df.groupby("target")["mcc"].agg(["mean", "std", "min", "max"])
    per_seed = df.groupby("seed")["mcc"].mean()
    nfeat_chg = df.groupby("target")["n_features"].nunique().gt(1).sum()
    print(f"  {pool:<6} targets={len(pm):3d}  pool-mean MCC={df['mcc'].mean():.4f} "
          f"+/- {df.groupby('target')['mcc'].mean().std():.4f}  "
          f"per-target std mean={pm['std'].mean():.4f} max={pm['std'].max():.4f} "
          f"({pm['std'].idxmax()})  n_features varies on {nfeat_chg}/{len(pm)}")
    print("          per-seed mean MCC: "
          + "  ".join(f"{s}:{m:.4f}" for s, m in sorted(per_seed.items())))


