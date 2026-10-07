"""Regenerate the sensitivity-analysis CSVs in ``sensitivity/results/``.

Why this script exists
----------------------
``AEFS_Sensitivity_Analysis.ipynb`` originally (a) derived the AEEEM defect
label from the ``numberOfBugsFoundUntil:`` *feature* instead of the ``class``
ground-truth column, and (b) ran a 3-stage variant of AEFS (RUP -> SHAP ->
stability, no SFS, no SMOTE) that does not match the 4-stage pipeline used
for the headline results in ``results/*_stage4_full_results.csv``.

This runner reproduces the notebook's four experiments -- SHAP-threshold
sweep, stability-threshold sweep, per-method runtime, and stage-wise feature
convergence -- with the *correct* loaders and the *full* AEFS pipeline
(RUP -> SFS -> SHAP -> stability, SMOTE on source projects only), reading
from the repository's ``data/`` directory so it runs outside Kaggle.

Outputs (same filenames/schema as before, plus ``after_sfs`` in the
convergence files):
    sensitivity/results/{aeeem,tera}_sensitivity_shap.csv
    sensitivity/results/{aeeem,tera}_sensitivity_stability.csv
    sensitivity/results/{aeeem,tera}_runtime.csv
    sensitivity/results/{aeeem,tera}_convergence.csv

Usage:
    python scripts/regen_sensitivity.py --pools aeeem tera
    python scripts/regen_sensitivity.py --pools aeeem --experiments shap convergence
"""

from __future__ import annotations

import argparse
import math
import re
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.feature_selection import RFE, SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

from lightgbm import LGBMClassifier  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

ROOT = Path(__file__).resolve().parents[1]
RAW_DIRS = {"aeeem": ROOT / "data" / "aeeem", "tera": ROOT / "data" / "tera"}
RESULTS_DIR = ROOT / "sensitivity" / "results"

RANDOM_STATE = 42

LGBM_SFS_PARAMS = dict(
    n_estimators=60, max_depth=5, num_leaves=15,
    verbosity=-1, n_jobs=-1,
)
LGBM_FINAL_PARAMS = dict(
    n_estimators=300, max_depth=-1, num_leaves=31, learning_rate=0.05,
    verbosity=-1, n_jobs=-1, random_state=RANDOM_STATE,
)
RFE_ESTIMATOR_PARAMS = dict(
    n_estimators=60, max_depth=5, num_leaves=15, verbosity=-1, n_jobs=-1,
)


# --------------------------------------------------------------------------
# SHAP -- shap.TreeExplainer with a LightGBM model is exact TreeSHAP, which is
# also what LightGBM's own `pred_contrib` returns.  Use shap when it imports
# (Kaggle / normal installs); fall back to pred_contrib when the local
# environment blocks shap's optional numba dependency.
# --------------------------------------------------------------------------
try:
    import shap as _shap
    _HAVE_SHAP = True
except Exception:  # pragma: no cover
    _shap = None
    _HAVE_SHAP = False


def tree_shap_values(clf, X: pd.DataFrame) -> np.ndarray:
    if _HAVE_SHAP:
        values = _shap.TreeExplainer(clf).shap_values(X)
        if isinstance(values, list):
            values = values[1]
        if getattr(values, "ndim", 0) == 3:
            values = values[:, :, 1]
        return np.asarray(values)
    contrib = clf.predict(X, pred_contrib=True)
    return np.asarray(contrib)[:, :-1]


# --------------------------------------------------------------------------
# Stage 1 -- loaders (mirrors AEFS_Pipeline_Friend.ipynb cells 9 and 11)
# --------------------------------------------------------------------------
TERA_TARGET_COL = "bug"
TERA_ID_LIKE_COLS = {"name", "name.1", "version"}
AEEEM_TARGET_COL = "class"
AEEEM_ID_LIKE_COLS = {"id"}
PROJECT_NAMES = {"eq": "equinox", "jdt": "jdt", "lc": "lucene", "ml": "mylyn", "pde": "pde"}


def _sanitize_column_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "", name)


def _assert_consistent_schema(cleaned: dict, pool_name: str) -> None:
    schemas = {name: tuple(df.columns) for name, df in cleaned.items()}
    if len(set(schemas.values())) > 1:
        raise ValueError(f"[{pool_name}] inconsistent column schema across projects")


def clean_tera_project(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if c.strip().lower() in TERA_ID_LIKE_COLS],
                 errors="ignore")
    if TERA_TARGET_COL not in df.columns:
        raise ValueError(f"missing '{TERA_TARGET_COL}'")
    feature_cols = [c for c in df.columns if c != TERA_TARGET_COL]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TERA_TARGET_COL] = pd.to_numeric(df[TERA_TARGET_COL], errors="coerce")
    df = df.dropna(axis=0, how="any")
    df[TERA_TARGET_COL] = (df[TERA_TARGET_COL] > 0).astype(int)
    df = df.rename(columns={TERA_TARGET_COL: "defective"})
    return df.drop_duplicates().reset_index(drop=True)


def clean_aeeem_project(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if c.strip().lower() in AEEEM_ID_LIKE_COLS],
                 errors="ignore")
    df = df.rename(columns={c: _sanitize_column_name(c) for c in df.columns})
    if AEEEM_TARGET_COL not in df.columns:
        raise ValueError(f"missing '{AEEEM_TARGET_COL}'")
    feature_cols = [c for c in df.columns if c != AEEEM_TARGET_COL]
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[AEEEM_TARGET_COL] = (df[AEEEM_TARGET_COL].astype(str).str.strip().str.lower()
                            .map({"buggy": 1, "clean": 0}))
    df = df.dropna(axis=0, how="any")
    df[AEEEM_TARGET_COL] = df[AEEEM_TARGET_COL].astype(int)
    df = df.rename(columns={AEEEM_TARGET_COL: "defective"})
    return df.drop_duplicates().reset_index(drop=True)


def _load_pool(raw_dir: Path, cleaner, pool_name: str,
               min_rows: int = 20, min_defective: int = 2) -> dict:
    cleaned, dropped = {}, {}
    for f in sorted(raw_dir.glob("*.csv")):
        stem = f.stem.lower()
        name = PROJECT_NAMES.get(stem, stem) if pool_name == "aeeem" else stem
        df = pd.read_csv(f)
        try:
            c = cleaner(df)
        except ValueError as e:
            dropped[name] = str(e)
            continue
        n_pos = int(c["defective"].sum())
        if len(c) < min_rows or n_pos < min_defective or n_pos == len(c):
            dropped[name] = f"rows={len(c)}, defective={n_pos}"
            continue
        cleaned[name] = c
    _assert_consistent_schema(cleaned, pool_name)
    print(f"[{pool_name}] {len(cleaned)} projects kept, {len(dropped)} dropped: "
          f"{sorted(dropped)}")
    return cleaned


def load_aeeem() -> dict:
    return _load_pool(RAW_DIRS["aeeem"], clean_aeeem_project, "aeeem")


def load_tera() -> dict:
    return _load_pool(RAW_DIRS["tera"], clean_tera_project, "tera")


# --------------------------------------------------------------------------
# Stages 2A-2D -- mirrors AEFS_Pipeline_Friend.ipynb cells 16/18/20/22
# --------------------------------------------------------------------------
def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int | None = None):
    if random_state is None:
        random_state = RANDOM_STATE
    minority_count = y.value_counts().min()
    if minority_count < 2:
        return X, y
    k_neighbors = min(5, minority_count - 1)
    return SMOTE(random_state=random_state, k_neighbors=k_neighbors).fit_resample(X, y)


def _relevance_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    scores = {}
    for col in X.columns:
        try:
            r, _ = pointbiserialr(y, X[col])
        except Exception:
            r = 0.0
        scores[col] = abs(r) if not np.isnan(r) else 0.0
    return pd.Series(scores)


def compute_rup_features(X: pd.DataFrame, y: pd.Series, var_threshold: float = 1e-4,
                         relevance_threshold: float = 0.02, redundancy_threshold: float = 0.90,
                         min_features: int = 5) -> list:
    all_features = list(X.columns)
    vt = VarianceThreshold(threshold=var_threshold)
    vt.fit(X)
    kept_after_variance = [c for c, k in zip(all_features, vt.get_support()) if k]
    working = X[kept_after_variance]
    relevance = _relevance_scores(working, y)
    kept_after_relevance = relevance[relevance >= relevance_threshold].index.tolist()

    if len(kept_after_relevance) > 1:
        working2 = working[kept_after_relevance]
        corr_matrix = working2.corr().abs()
        ordered = relevance.loc[kept_after_relevance].sort_values(ascending=False).index.tolist()
        survivors, removed = [], set()
        for feat in ordered:
            if feat in removed:
                continue
            survivors.append(feat)
            for other in ordered:
                if other == feat or other in removed or other in survivors:
                    continue
                if corr_matrix.loc[feat, other] >= redundancy_threshold:
                    removed.add(other)
        selected = survivors
    else:
        selected = kept_after_relevance

    if len(selected) < min_features:
        remaining_ranked = relevance.drop(index=selected, errors="ignore").sort_values(ascending=False)
        selected = selected + remaining_ranked.head(min_features - len(selected)).index.tolist()
    return selected


def _cv_mcc(X: pd.DataFrame, y: pd.Series, k: int = 3, random_state: int | None = None) -> float:
    if random_state is None:
        random_state = RANDOM_STATE
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    scores = []
    for tr, va in skf.split(X, y):
        clf = LGBMClassifier(**LGBM_SFS_PARAMS, random_state=random_state)
        clf.fit(X.iloc[tr], y.iloc[tr])
        scores.append(matthews_corrcoef(y.iloc[va], clf.predict(X.iloc[va])))
    return float(np.mean(scores))


def sequential_forward_selection(X: pd.DataFrame, y: pd.Series, candidates: list,
                                 cv_folds: int = 3, min_improvement: float = 0.001,
                                 min_features: int = 5,
                                 random_state: int | None = None) -> list:
    if random_state is None:
        random_state = RANDOM_STATE
    remaining, selected = list(candidates), []
    best_score = -1.0
    while remaining:
        scores = {f: _cv_mcc(X[selected + [f]], y, k=cv_folds, random_state=random_state)
                  for f in remaining}
        best_feat = max(scores, key=scores.get)
        best_trial = scores[best_feat]
        if best_trial - best_score < min_improvement and selected:
            break
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_score = best_trial

    if len(selected) < min_features and remaining:
        scores_fb = {f: _cv_mcc(X[selected + [f]], y, k=cv_folds, random_state=random_state)
                     for f in remaining}
        while len(selected) < min_features and remaining:
            best_fb = max(scores_fb, key=scores_fb.get)
            selected.append(best_fb)
            remaining.remove(best_fb)
            scores_fb.pop(best_fb)
            for f in list(scores_fb):
                scores_fb[f] = _cv_mcc(X[selected + [f]], y, k=cv_folds, random_state=random_state)
    return selected


def shap_refine(X: pd.DataFrame, y: pd.Series, candidates: list,
                min_contribution_frac: float = 0.02, min_features: int = 5) -> list:
    if len(candidates) <= min_features:
        return list(candidates)
    clf = LGBMClassifier(**LGBM_SFS_PARAMS, random_state=RANDOM_STATE)
    clf.fit(X[candidates], y)
    mean_abs = np.abs(tree_shap_values(clf, X[candidates])).mean(axis=0)
    importance = pd.Series(mean_abs, index=candidates).sort_values(ascending=False)
    total = importance.sum()
    frac = importance / total if total > 0 else importance
    keep = frac[frac >= min_contribution_frac].index.tolist()
    if len(keep) < min_features:
        keep = importance.head(min_features).index.tolist()
    return keep


def _jaccard(a: set, b: set) -> float:
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def stability_check(X_presmote: pd.DataFrame, y_presmote: pd.Series,
                    project_labels: pd.Series, candidates: list,
                    stability_threshold: float = 0.30, top_k_fraction: float = 0.7,
                    min_project_rows: int = 20, min_project_minority: int = 2,
                    min_features: int = 5) -> tuple:
    top_k = max(1, math.ceil(top_k_fraction * len(candidates)))
    project_top_sets, skipped = {}, 0
    for proj in project_labels.unique():
        mask = project_labels == proj
        Xp = X_presmote.loc[mask, candidates]
        yp = y_presmote.loc[mask]
        if len(Xp) < min_project_rows or yp.value_counts().min() < min_project_minority:
            skipped += 1
            continue
        clf = LGBMClassifier(**LGBM_SFS_PARAMS, random_state=RANDOM_STATE)
        clf.fit(Xp, yp)
        mean_abs = np.abs(tree_shap_values(clf, Xp)).mean(axis=0)
        ranked = pd.Series(mean_abs, index=candidates).sort_values(ascending=False)
        project_top_sets[proj] = set(ranked.head(top_k).index)

    if not project_top_sets:
        return list(candidates), 1.0

    presence = {f: sum(f in s for s in project_top_sets.values()) / len(project_top_sets)
                for f in candidates}
    selected = [f for f in candidates if presence[f] >= stability_threshold]
    if len(selected) < min_features:
        selected = sorted(candidates, key=lambda f: presence[f], reverse=True)[:min_features]

    names = list(project_top_sets)
    if len(names) >= 2:
        jac = float(np.mean([_jaccard(project_top_sets[a], project_top_sets[b])
                             for a, b in combinations(names, 2)]))
    else:
        jac = 1.0
    return selected, jac


# --------------------------------------------------------------------------
# LOPO fold generation (source-only scaling / RUP / SMOTE, never the target)
# --------------------------------------------------------------------------
def generate_lopo_folds(pool: dict) -> list:
    names = list(pool.keys())
    folds = []
    for target_name in names:
        test_df = pool[target_name]
        source_names = [n for n in names if n != target_name]
        source_dfs = [pool[n] for n in source_names]
        train_df = pd.concat(source_dfs, axis=0, ignore_index=True)
        if test_df["defective"].sum() < 1:
            continue
        project_labels = pd.Series(
            np.concatenate([[n] * len(df) for n, df in zip(source_names, source_dfs)]),
            index=train_df.index,
        )
        feature_cols = [c for c in train_df.columns if c != "defective"]
        X_raw, y_raw = train_df[feature_cols], train_df["defective"]
        X_te_raw, y_te = test_df[feature_cols], test_df["defective"]

        scaler = MinMaxScaler()
        X_sc = pd.DataFrame(scaler.fit_transform(X_raw), columns=feature_cols, index=X_raw.index)
        X_te = pd.DataFrame(scaler.transform(X_te_raw), columns=feature_cols, index=X_te_raw.index)

        selected = compute_rup_features(X_sc, y_raw)
        X_all, X_te_all = X_sc.copy(), X_te.copy()
        X_sc, X_te = X_sc[selected], X_te[selected]
        X_presmote, y_presmote = X_sc.copy(), y_raw.copy()
        X_tr, y_tr = apply_smote(X_sc, y_raw)

        folds.append(dict(target=target_name, X_train=X_tr, y_train=y_tr,
                          X_test=X_te, y_test=y_te, X_all=X_all, X_te_all=X_te_all,
                          X_presmote=X_presmote, y_presmote=y_presmote,
                          project_labels=project_labels, rup_candidates=selected))
    return folds


def _evaluate(X_tr, y_tr, X_te, y_te, feats) -> dict:
    clf = LGBMClassifier(**LGBM_FINAL_PARAMS)
    clf.fit(X_tr[feats], y_tr)
    proba = clf.predict_proba(X_te[feats])[:, 1]
    pred = (proba >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_te, proba)
    except ValueError:
        auc = float("nan")
    return dict(n_features=len(feats),
                mcc=round(matthews_corrcoef(y_te, pred), 4),
                f1=round(f1_score(y_te, pred, zero_division=0), 4),
                auc=round(auc, 4))


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def run_shap_sensitivity(folds: list, pool_name: str,
                         thresholds=(0.01, 0.02, 0.03, 0.05)) -> pd.DataFrame:
    rows = []
    for i, fold in enumerate(tqdm(folds, desc=f"[{pool_name}] SHAP sweep"), 1):
        sfs_feats = sequential_forward_selection(fold["X_train"], fold["y_train"],
                                                 fold["rup_candidates"])
        for thr in thresholds:
            shap_feats = shap_refine(fold["X_train"], fold["y_train"], sfs_feats, thr)
            final_feats, _ = stability_check(fold["X_presmote"], fold["y_presmote"],
                                             fold["project_labels"], shap_feats, 0.30)
            m = _evaluate(fold["X_train"], fold["y_train"], fold["X_test"],
                          fold["y_test"], final_feats)
            rows.append(dict(target=fold["target"], shap_threshold=thr,
                             n_features=m["n_features"], mcc=m["mcc"],
                             f1=m["f1"], auc=m["auc"]))
        if i % 10 == 0 or i == len(folds):
            print(f"    fold {i}/{len(folds)}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{pool_name}_sensitivity_shap.csv", index=False, lineterminator="\n")
    print(f"[{pool_name}] SHAP sensitivity: saved {len(df)} rows")
    return df


def run_stability_sensitivity(folds: list, pool_name: str,
                              thresholds=(0.20, 0.30, 0.40, 0.50)) -> pd.DataFrame:
    rows = []
    for i, fold in enumerate(tqdm(folds, desc=f"[{pool_name}] stability sweep"), 1):
        sfs_feats = sequential_forward_selection(fold["X_train"], fold["y_train"],
                                                 fold["rup_candidates"])
        shap_feats = shap_refine(fold["X_train"], fold["y_train"], sfs_feats, 0.02)
        for thr in thresholds:
            final_feats, jac = stability_check(fold["X_presmote"], fold["y_presmote"],
                                               fold["project_labels"], shap_feats, thr)
            m = _evaluate(fold["X_train"], fold["y_train"], fold["X_test"],
                          fold["y_test"], final_feats)
            rows.append(dict(target=fold["target"], stability_threshold=thr,
                             n_features=m["n_features"], mean_jaccard=round(jac, 4),
                             mcc=m["mcc"], f1=m["f1"], auc=m["auc"]))
        if i % 10 == 0 or i == len(folds):
            print(f"    fold {i}/{len(folds)}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{pool_name}_sensitivity_stability.csv", index=False, lineterminator="\n")
    print(f"[{pool_name}] Stability sensitivity: saved {len(df)} rows")
    return df


def measure_runtime(folds: list, pool_name: str, n_folds: int | None = None) -> pd.DataFrame:
    methods = ["AEFS", "AllFeatures", "FilterMI", "WrapperRFE"]
    timings = {m: [] for m in methods}
    if n_folds:
        folds = folds[:n_folds]
    print(f"[{pool_name}] Runtime measurement: {len(folds)} folds")

    for i, fold in enumerate(tqdm(folds, desc=f"[{pool_name}] runtime"), 1):
        # Baseline training sets are built outside the timed region so the
        # clock measures feature selection + model fit + predict only.
        X_all_sm, y_all_sm = apply_smote(fold["X_all"], fold["y_presmote"])
        X_te_all = fold["X_te_all"]

        t0 = time.perf_counter()
        candidates = compute_rup_features(fold["X_all"], fold["y_presmote"])
        sfs_feats = sequential_forward_selection(fold["X_train"], fold["y_train"], candidates)
        shap_feats = shap_refine(fold["X_train"], fold["y_train"], sfs_feats, 0.02)
        final_feats, _ = stability_check(fold["X_presmote"], fold["y_presmote"],
                                         fold["project_labels"], shap_feats, 0.30)
        k = len(final_feats)
        lgbm = LGBMClassifier(**LGBM_FINAL_PARAMS)
        lgbm.fit(fold["X_train"][final_feats], fold["y_train"])
        lgbm.predict(fold["X_test"][final_feats])
        timings["AEFS"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        lgbm = LGBMClassifier(**LGBM_FINAL_PARAMS)
        lgbm.fit(X_all_sm, y_all_sm)
        lgbm.predict(X_te_all)
        timings["AllFeatures"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        k_mi = min(k, X_all_sm.shape[1])
        sel = SelectKBest(score_func=lambda X_, y_: mutual_info_classif(
            X_, y_, random_state=RANDOM_STATE), k=k_mi)
        sel.fit(X_all_sm, y_all_sm)
        feats = list(X_all_sm.columns[sel.get_support()])
        lgbm = LGBMClassifier(**LGBM_FINAL_PARAMS)
        lgbm.fit(X_all_sm[feats], y_all_sm)
        lgbm.predict(X_te_all[feats])
        timings["FilterMI"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        k_rfe = min(k, X_all_sm.shape[1])
        rfe = RFE(LGBMClassifier(**RFE_ESTIMATOR_PARAMS, random_state=RANDOM_STATE),
                  n_features_to_select=k_rfe, step=0.2)
        rfe.fit(X_all_sm, y_all_sm)
        feats = list(X_all_sm.columns[rfe.support_])
        lgbm = LGBMClassifier(**LGBM_FINAL_PARAMS)
        lgbm.fit(X_all_sm[feats], y_all_sm)
        lgbm.predict(X_te_all[feats])
        timings["WrapperRFE"].append(time.perf_counter() - t0)

        if i % 5 == 0 or i == len(folds):
            print(f"    fold {i}/{len(folds)}")

    rows = [dict(method=m, mean_time_s=round(float(np.mean(t)), 2),
                 std_time_s=round(float(np.std(t)), 2),
                 total_time_s=round(float(np.sum(t)), 2))
            for m, t in timings.items()]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{pool_name}_runtime.csv", index=False, lineterminator="\n")
    print(f"[{pool_name}] Runtime: saved\n{df.to_string(index=False)}")
    return df


def measure_convergence(folds: list, pool_name: str) -> pd.DataFrame:
    rows = []
    for i, fold in enumerate(tqdm(folds, desc=f"[{pool_name}] convergence"), 1):
        original = fold["X_all"].shape[1]
        candidates = compute_rup_features(fold["X_all"], fold["y_presmote"])
        sfs_feats = sequential_forward_selection(fold["X_train"], fold["y_train"], candidates)
        shap_feats = shap_refine(fold["X_train"], fold["y_train"], sfs_feats, 0.02)
        final_feats, _ = stability_check(fold["X_presmote"], fold["y_presmote"],
                                         fold["project_labels"], shap_feats, 0.30)
        rows.append(dict(target=fold["target"], original=original,
                         after_rup=len(candidates), after_sfs=len(sfs_feats),
                         after_shap=len(shap_feats), after_stability=len(final_feats)))
        if i % 10 == 0 or i == len(folds):
            print(f"    fold {i}/{len(folds)}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"{pool_name}_convergence.csv", index=False, lineterminator="\n")
    print(f"[{pool_name}] Convergence: saved\n{df.mean(numeric_only=True).round(1).to_string()}")
    return df


DEFAULT_SEEDS = [42, 7, 123, 2024, 555]


def run_multiseed(pools: list, seeds: list, shard: int = 0,
                  nshards: int = 1) -> pd.DataFrame:
    """Multi-seed variance of the AEFS pipeline at the default thresholds.

    The seed drives SMOTE, the SFS cross-validation, the per-project models of
    Stage 2D and the final classifier, so the resulting spread is the variance
    of the whole pipeline rather than of one fit.  Folds are regenerated per
    seed (SMOTE is part of fold construction).  `shard`/`nshards` split the
    fold list so several workers can cover the same seed set.
    """
    global RANDOM_STATE
    loaders = {"aeeem": load_aeeem, "tera": load_tera}
    saved_seed, saved_final = RANDOM_STATE, dict(LGBM_FINAL_PARAMS)
    rows = []
    try:
        for pool in pools:
            for seed in seeds:
                RANDOM_STATE = seed
                LGBM_FINAL_PARAMS["random_state"] = seed
                folds = generate_lopo_folds(loaders[pool]())
                mine = [f for i, f in enumerate(folds) if i % nshards == shard]
                if not mine:
                    continue
                print(f"[{pool}] seed {seed}: {len(mine)}/{len(folds)} folds", flush=True)
                for fold in tqdm(mine, desc=f"[{pool}] seed={seed}"):
                    sfs_feats = sequential_forward_selection(
                        fold["X_train"], fold["y_train"], fold["rup_candidates"])
                    shap_feats = shap_refine(fold["X_train"], fold["y_train"],
                                             sfs_feats, 0.02)
                    final_feats, jac = stability_check(
                        fold["X_presmote"], fold["y_presmote"],
                        fold["project_labels"], shap_feats, 0.30)
                    m = _evaluate(fold["X_train"], fold["y_train"],
                                  fold["X_test"], fold["y_test"], final_feats)
                    rows.append(dict(pool=pool, seed=seed, target=fold["target"],
                                     n_features=m["n_features"],
                                     mean_jaccard=round(float(jac), 4),
                                     mcc=m["mcc"], f1=m["f1"], auc=m["auc"]))
    finally:
        RANDOM_STATE = saved_seed
        LGBM_FINAL_PARAMS.clear()
        LGBM_FINAL_PARAMS.update(saved_final)

    df = pd.DataFrame(rows, columns=["pool", "seed", "target", "n_features",
                                     "mean_jaccard", "mcc", "f1", "auc"])
    for pool in pools:
        part = df[df["pool"] == pool]
        if part.empty:
            continue
        dest = RESULTS_DIR / f"{pool}_multiseed_s{shard:02d}.csv"
        part.drop(columns=["pool"]).to_csv(dest, index=False, lineterminator="\n")
        print(f"[{pool}] multi-seed: saved {len(part)} rows -> {dest.name}")
    return df


EXPERIMENTS = {
    "shap": lambda folds, pool: run_shap_sensitivity(folds, pool),
    "stability": lambda folds, pool: run_stability_sensitivity(folds, pool),
    "runtime": lambda folds, pool: measure_runtime(folds, pool, n_folds=10 if pool == "tera" else None),
    "convergence": lambda folds, pool: measure_convergence(folds, pool),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", nargs="+", default=["aeeem", "tera"], choices=["aeeem", "tera"])
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS),
                    choices=list(EXPERIMENTS) + ["multiseed"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
                    help="seeds for --experiments multiseed")
    ap.add_argument("--shard", type=int, default=0,
                    help="fold shard index for --experiments multiseed")
    ap.add_argument("--nshards", type=int, default=1,
                    help="number of fold shards for --experiments multiseed")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"SHAP backend: {'shap.TreeExplainer' if _HAVE_SHAP else 'lightgbm pred_contrib'}")

    if "multiseed" in args.experiments:
        run_multiseed(args.pools, args.seeds, args.shard, args.nshards)
        return

    loaders = {"aeeem": load_aeeem, "tera": load_tera}
    for pool in args.pools:
        print("=" * 70)
        data = loaders[pool]()
        folds = generate_lopo_folds(data)
        print(f"[{pool}] {len(folds)} LOPO folds ready")
        for exp in args.experiments:
            EXPERIMENTS[exp](folds, pool)


if __name__ == "__main__":
    main()
