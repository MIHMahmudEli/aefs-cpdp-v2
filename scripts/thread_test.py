"""Is the local-vs-Kaggle MCC gap just LightGBM thread-count nondeterminism?

n_jobs=-1 gives 8 threads locally and 4 on Kaggle.  Fit the same data with
different thread counts and see whether the predictions move.

Run:  python scripts/thread_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from sklearn.metrics import matthews_corrcoef, f1_score, roc_auc_score

import regen_sensitivity as rs
from lightgbm import LGBMClassifier

rs.RAW_DIRS = {"aeeem": rs.RAW_DIRS["aeeem"], "tera": rs.RAW_DIRS["tera"]}
data = rs.load_aeeem()
folds = rs.generate_lopo_folds(data)

for fold in folds:
    Xtr, ytr = fold["X_train"], fold["y_train"]
    Xte, yte = fold["X_test"], fold["y_test"]
    rows = []
    for jobs in (4, 8, 4, 8):
        clf = LGBMClassifier(**{**rs.LGBM_FINAL_PARAMS, "n_jobs": jobs})
        clf.fit(Xtr, ytr)
        p = clf.predict(Xte)
        proba = clf.predict_proba(Xte)[:, 1]
        rows.append((jobs, matthews_corrcoef(yte, p),
                     f1_score(yte, p, zero_division=0),
                     roc_auc_score(yte, proba)))
    df = pd.DataFrame(rows, columns=["n_jobs", "mcc", "f1", "auc"])
    uniq = df.drop_duplicates(subset=["mcc", "f1", "auc"])
    print(f"{fold['target']:<12} " +
          "  ".join(f"j{j}:MCC={m:.4f}" for j, m, _, _ in rows) +
          f"   distinct={len(uniq)}")


