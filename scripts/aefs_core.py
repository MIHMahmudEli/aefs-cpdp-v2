"""Canonical AEFS implementation - single source of truth for every number.

Why this module exists
----------------------
Before it, four different pipelines existed in the repository:

* ``AEFS_Pipeline.ipynb``            4 stages + SMOTE          (AEEEM, TeraPromise)
* ``AEFS_ReLink_NASA_Expansion.ipynb`` 3 stages, no SFS, no SMOTE (NASA, ReLink)
* ``AEFS_Extended_Eval.ipynb``       4 stages, no SMOTE, different AEEEM loader,
                                     14-project TeraPromise subsample
* ``scripts/regen_sensitivity.py``   4 stages + SMOTE (sensitivity sweeps only)

so the four result tables were not protocol-comparable (audit finding A1-A4 in
``docs/EXPERIMENTS.md``).  Everything below implements *one* protocol:

    source-only MinMax scaling
      -> Stage 2A RUP
      -> SMOTE on the source projects only          (every method)
      -> Stage 2B SFS -> Stage 2C SHAP -> Stage 2D stability   (AEFS)
      -> final classifier on the held-out target project

Baselines (AllFeatures / FilterMI / WrapperRFE / BorutaSHAP / CPDP-aware
filters) start from the *pre-RUP* feature space - RUP is part of AEFS itself -
and re-apply SMOTE on the columns they select, so no method is advantaged.

The stage functions are imported from ``regen_sensitivity.py`` where they
already exist, so the sensitivity sweeps and the headline tables can never
drift apart again.

Usage
-----
    python scripts/run_experiment.py --experiment main --pools aeeem
"""

from __future__ import annotations

import math
import re
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import regen_sensitivity as rs  # noqa: E402  (stage functions + aeeem/tera loaders)

from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW_DIRS = {
    "aeeem": ROOT / "data" / "aeeem",
    "tera": ROOT / "data" / "tera",
    "nasa": ROOT / "data" / "nasa",
    "relink": ROOT / "data" / "relink",
}


def _extract_zip(here: Path) -> Path | None:
    """Kaggle datasets upload sub-folders as <folder>.zip (the CLI cannot
    publish a directory tree), so unzip it into a writable location."""
    z = here / "data.zip"
    if not z.is_file():
        return None
    import zipfile
    for dest in (Path("/kaggle/working/data"), Path.cwd() / "data",
                 here / "_data"):
        try:
            dest.mkdir(parents=True, exist_ok=True)
            probe = dest / ".write_probe"
            probe.write_text("ok", encoding="ascii")
            probe.unlink()
        except OSError:
            continue
        if (dest / "aeeem").is_dir():
            return dest
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
        except (OSError, zipfile.BadZipFile) as exc:
            print(f"[data] extraction to {dest} failed: {exc}", flush=True)
            continue
        if (dest / "aeeem").is_dir():
            print(f"[data] extracted {z} -> {dest}", flush=True)
            return dest
    return None


def resolve_data_root() -> Path | None:
    """Find the directory holding aeeem/ tera/ nasa/ relink/.

    Locally that is <repo>/data; on Kaggle it is the mounted assets dataset
    (<assets>/data, published as data.zip).  Called at import time so a
    Kaggle kernel never silently loads an empty pool (audit: the first
    validation run produced 0 folds because ``regen_sensitivity.ROOT``
    pointed at /kaggle/input).
    """
    here = Path(__file__).resolve().parent
    if not (here / "data" / "aeeem").is_dir():
        _extract_zip(here)
    candidates = [ROOT / "data", here / "data", here.parent / "data",
                  Path("/kaggle/working/data"), Path.cwd() / "data",
                  Path("/kaggle/input")]
    for base in list(candidates):
        if base.is_dir():
            try:
                candidates += [d for d in base.iterdir() if d.is_dir()]
            except OSError:
                pass
    for c in candidates:
        if (c / "aeeem").is_dir() and (c / "tera").is_dir():
            return c
    return None


_DATA_ROOT = resolve_data_root()
if _DATA_ROOT is not None:
    RAW_DIRS = {name: _DATA_ROOT / name for name in
                ("aeeem", "tera", "nasa", "relink")}
    # regen_sensitivity loads aeeem/tera through its own RAW_DIRS, whose ROOT
    # is derived from __file__ and points at the wrong place inside a Kaggle
    # kernel, so keep both in sync.
    rs.RAW_DIRS = {"aeeem": RAW_DIRS["aeeem"], "tera": RAW_DIRS["tera"]}
POOL_LABELS = {"aeeem": "AEEEM", "tera": "TeraPromise",
               "nasa": "NASA MDP", "relink": "ReLink"}

DEFAULT_SEEDS = [42, 7, 123, 2024, 555]

# The seed and the final-model hyper-parameters are deliberately aliased to
# regen_sensitivity's globals: a multi-seed run mutates them in one place and
# every consumer (sensitivity sweeps, headline grid) sees the same value.
RANDOM_STATE = rs.RANDOM_STATE
LGBM_SFS_PARAMS = dict(n_estimators=60, max_depth=5, num_leaves=15,
                       verbosity=-1, n_jobs=-1)
LGBM_FINAL_PARAMS = rs.LGBM_FINAL_PARAMS
RF_FINAL_PARAMS = dict(n_estimators=300, max_depth=None, n_jobs=-1,
                       random_state=RANDOM_STATE)
SVM_FINAL_PARAMS = dict(kernel="rbf", probability=True, random_state=RANDOM_STATE)
RFE_ESTIMATOR_PARAMS = dict(n_estimators=60, max_depth=5, num_leaves=15,
                            verbosity=-1, n_jobs=-1)

MAIN_METHODS = ["AEFS", "Ablation_no2D", "AllFeatures", "FilterMI",
                "WrapperRFE", "BorutaSHAP"]
ABLATION_METHODS = ["Ablation_noRUP", "Ablation_noSFS", "Ablation_noSHAP"]
CPDP_METHODS = ["MI_DS", "MI_TW"]
SENS_METHODS = ["AEFS", "AllFeatures", "FilterMI", "WrapperRFE"]
SENS_SHAP_GRID = (0.01, 0.02, 0.03, 0.05)
SENS_STABILITY_GRID = (0.20, 0.30, 0.40, 0.50)
SENS_COLUMNS = ["pool", "target", "experiment", "setting", "method",
                "model", "seed", "n_features", "mean_jaccard",
                "runtime_s", "test_mcc", "test_f1", "test_auc_roc"]
MAIN_MODELS = ["lightgbm", "random_forest", "svm"]
ABLATION_MODELS = ["lightgbm", "random_forest"]

# TeraPromise LOPO trains on ~86k source rows (70 projects), ~128k after SMOTE.
# One RBF-SVC fit with Platt calibration then takes 40-70 min on Kaggle's 4
# CPU cores, so 6 methods x 5 folds would blow through the 12 h session limit
# (measured locally: SVC 20k/40k/86k rows -> 73/278/1500+ s).  The legacy
# stage-4 tables made exactly the same call - ``tera_stage4_full_results.csv``
# has no svm rows - so TeraPromise is evaluated with LightGBM and RandomForest
# only, and the manuscript states that SVM is reported for the three pools
# whose training sets it can finish.
POOL_MODELS = {
    "aeeem": MAIN_MODELS,
    "nasa": MAIN_MODELS,
    "relink": MAIN_MODELS,
    "tera": ["lightgbm", "random_forest"],
}


def models_for(pool: str) -> list:
    return list(POOL_MODELS.get(pool, MAIN_MODELS))

# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
NASA_TARGET_CANDIDATES = ["Defective", "c", "defects"]
NASA_TARGET_MAP = {"Y": 1, "N": 0, "TRUE": 1, "FALSE": 0,
                   "TRUE ": 1, "FALSE ": 0, "TRUE\n": 1, "FALSE\n": 0,
                   "1": 1, "0": 0, "T": 1, "F": 0, "YES": 1, "NO": 0}
NASA_EXCLUDE = {"pc1", "pc5", "pc6"}   # incompatible / absent schemas
RELINK_TARGET = "isDefective"


def load_nasa(min_rows: int = 20, min_defective: int = 2) -> dict:
    """NASA MDP: PC3, PC4, cm1, mw1 (PC1 excluded - 22-column schema)."""
    raw = RAW_DIRS["nasa"]
    cleaned, dropped = {}, {}
    for f in sorted(raw.glob("*.csv")):
        name = f.stem.lower()
        if name in NASA_EXCLUDE:
            dropped[name] = "excluded schema"
            continue
        df = pd.read_csv(f)
        target = next((c for c in NASA_TARGET_CANDIDATES if c in df.columns), None)
        if target is None:
            dropped[name] = "no target column"
            continue
        y = (df[target].astype(str).str.strip().str.upper()
             .map(NASA_TARGET_MAP).astype("float"))
        if y.isna().any():
            y = df[target].astype(str).str.strip().map(
                lambda v: 1.0 if str(v).lower() in ("1", "true", "y", "yes")
                else (0.0 if str(v).lower() in ("0", "false", "n", "no")
                      else np.nan))
        df = df.drop(columns=[target])
        num = df.apply(pd.to_numeric, errors="coerce")
        keep = [c for c in num.columns if not num[c].isna().any()]
        out = num[keep].copy()
        out["defective"] = y.values
        out = out.dropna().drop_duplicates().reset_index(drop=True)
        out["defective"] = out["defective"].astype(int)
        if len(out) >= min_rows and out["defective"].sum() >= min_defective \
                and out["defective"].nunique() == 2:
            cleaned[name] = out
        else:
            dropped[name] = f"rows={len(out)}, defective={int(out['defective'].sum())}"
    rs._assert_consistent_schema(cleaned, "nasa")
    print(f"[nasa] {len(cleaned)} projects kept, {len(dropped)} dropped: {sorted(dropped)}")
    return cleaned


def load_relink(min_rows: int = 20, min_defective: int = 2) -> dict:
    raw = RAW_DIRS["relink"]
    cleaned, dropped = {}, {}
    for f in sorted(raw.glob("*.csv")):
        name = f.stem.lower()
        df = pd.read_csv(f)
        if RELINK_TARGET not in df.columns:
            dropped[name] = "no target column"
            continue
        df["defective"] = df[RELINK_TARGET].map({"buggy": 1, "clean": 0})
        df = df.drop(columns=[RELINK_TARGET])
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        df["defective"] = df["defective"].astype(int)
        if len(df) >= min_rows and df["defective"].sum() >= min_defective \
                and df["defective"].nunique() == 2:
            cleaned[name] = df
        else:
            dropped[name] = f"rows={len(df)}, defective={int(df['defective'].sum())}"
    rs._assert_consistent_schema(cleaned, "relink")
    print(f"[relink] {len(cleaned)} projects kept, {len(dropped)} dropped: {sorted(dropped)}")
    return cleaned


def load_pool(pool: str, min_rows: int = 20, min_defective: int = 2) -> dict:
    if pool == "aeeem":
        return rs.load_aeeem()
    if pool == "tera":
        return rs.load_tera()
    if pool == "nasa":
        return load_nasa(min_rows, min_defective)
    if pool == "relink":
        return load_relink(min_rows, min_defective)
    raise ValueError(f"unknown pool {pool!r}")


# --------------------------------------------------------------------------
# Folds (identical to AEFS_Pipeline.ipynb / regen_sensitivity.generate_lopo_folds)
# --------------------------------------------------------------------------
def generate_lopo_folds(pool: dict) -> list:
    return rs.generate_lopo_folds(pool)


def fold_fullspace(fold: dict) -> tuple:
    """Pre-RUP feature space: (X_train_all, y_presmote, X_test_all)."""
    return fold["X_all"], fold["y_presmote"], fold["X_te_all"]


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
def make_model(name: str):
    if name == "lightgbm":
        return LGBMClassifier(**LGBM_FINAL_PARAMS)
    if name == "random_forest":
        return RandomForestClassifier(**RF_FINAL_PARAMS)
    if name == "svm":
        return SVC(**SVM_FINAL_PARAMS)
    raise ValueError(f"unknown model {name!r}")


def score_predictions(y_true, y_pred, y_proba) -> dict:
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")
    return dict(
        test_mcc=round(float(matthews_corrcoef(y_true, y_pred)), 4),
        test_f1=round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        test_auc_roc=round(float(auc), 4),
    )


def fit_predict(model_name, X_tr, y_tr, X_te, feats) -> tuple:
    """Returns (y_pred, y_proba) for the requested classifier."""
    clf = make_model(model_name)
    clf.fit(X_tr[feats], y_tr)
    proba = clf.predict_proba(X_te[feats])[:, 1]
    return (proba >= 0.5).astype(int), proba


# --------------------------------------------------------------------------
# Feature-selection methods
# --------------------------------------------------------------------------
def select_filter_mi(X: pd.DataFrame, y: pd.Series, k: int) -> list:
    k = min(k, X.shape[1])
    sel = SelectKBest(score_func=lambda X_, y_: mutual_info_classif(
        X_, y_, random_state=rs.RANDOM_STATE), k=k)
    sel.fit(X, y)
    return list(X.columns[sel.get_support()])


def select_wrapper_rfe(X: pd.DataFrame, y: pd.Series, k: int) -> list:
    k = min(k, X.shape[1])
    rfe = RFE(LGBMClassifier(**RFE_ESTIMATOR_PARAMS, random_state=rs.RANDOM_STATE),
              n_features_to_select=k, step=0.2)
    rfe.fit(X, y)
    return list(X.columns[rfe.support_])


def select_borutashap(X: pd.DataFrame, y: pd.Series, max_iter: int = 10,
                      alpha: float = 0.05, min_features: int = 5) -> list:
    """BorutaSHAP: shadow-feature binomial test driven by SHAP importances.

    Shadow features are permutations of the real ones; a feature is confirmed
    when it beats the strongest shadow more often than chance (p < alpha).
    """
    if X.shape[1] <= min_features:
        return list(X.columns)
    rng = np.random.RandomState(rs.RANDOM_STATE)
    shadow = pd.DataFrame({f"shadow_{c}": rng.permutation(X[c].values)
                           for c in X.columns}, index=X.index)
    confirmed: set = set()
    tentative = set(X.columns)
    imp = None
    for _ in range(max_iter):
        if len(tentative) <= min_features or not tentative:
            break
        cols = sorted(confirmed | tentative) + list(shadow.columns)
        clf = LGBMClassifier(**LGBM_SFS_PARAMS, random_state=rs.RANDOM_STATE)
        clf.fit(pd.concat([X[sorted(confirmed | tentative)], shadow], axis=1), y)
        values = rs.tree_shap_values(
            clf, pd.concat([X[sorted(confirmed | tentative)], shadow], axis=1))
        imp = pd.Series(np.abs(values).mean(axis=0),
                        index=list(sorted(confirmed | tentative)) + list(shadow.columns))
        smax = imp[list(shadow.columns)].max()
        hits = {f: bool(imp.get(f, 0.0) > smax) for f in sorted(tentative)}
        n_hits, n_tot = sum(hits.values()), len(hits)
        if n_tot == 0:
            break
        p_value = float(1 - sps.binom.cdf(n_hits - 1, n_tot, 0.5))
        if p_value < alpha:
            confirmed |= {f for f, h in hits.items() if h}
            tentative -= confirmed
        tentative -= set(shadow.columns)
    result = sorted(confirmed | tentative)
    if len(result) < min_features and imp is not None:
        real = imp.drop(labels=[c for c in imp.index if c.startswith("shadow_")],
                        errors="ignore")
        result = list(real.nlargest(min_features).index)
    return result


def _distribution_similarity(x_src: pd.Series, x_tgt: pd.Series,
                             bins: int = 20) -> float:
    """Histogram-intersection overlap of the source and target marginals.

    Uses only feature values, so no target labels are required (legal in LOPO).
    """
    lo = float(min(x_src.min(), x_tgt.min()))
    hi = float(max(x_src.max(), x_tgt.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    hs, _ = np.histogram(x_src.values, bins=edges)
    ht, _ = np.histogram(x_tgt.values, bins=edges)
    hs = hs / max(hs.sum(), 1)
    ht = ht / max(ht.sum(), 1)
    return float(np.minimum(hs, ht).sum())


def select_mi_ds(X: pd.DataFrame, y: pd.Series, k: int,
                 X_target: pd.DataFrame) -> list:
    """CPDP-aware filter: mutual information x source-target similarity.

    Relevance (MI with the source labels) is reweighted by how well each
    feature's marginal distribution transfers to the held-out target project -
    the feature-level distribution-similarity principle used by CPDP feature
    selection methods (Khatri & Singh 2023; Lei et al. 2020).  Target labels
    are never used.
    """
    k = min(k, X.shape[1])
    mi = mutual_info_classif(X, y, random_state=rs.RANDOM_STATE)
    sim = np.array([_distribution_similarity(X[c], X_target[c]) for c in X.columns])
    score = mi * sim
    order = np.argsort(-score)
    return [X.columns[i] for i in order[:k]]


def select_mi_tw(X: pd.DataFrame, y: pd.Series, k: int,
                 X_target: pd.DataFrame) -> list:
    """CPDP-aware filter: MI computed on target-similar source instances."""
    k = min(k, X.shape[1])
    d = np.zeros(len(X))
    for c in X.columns:
        lo = float(min(X[c].min(), X_target[c].min()))
        hi = float(max(X[c].max(), X_target[c].max()))
        if hi <= lo:
            continue
        span = hi - lo
        d += np.abs(X[c].values - float(X_target[c].mean())) / span
    d = d / max(len(X.columns), 1)
    # exponential similarity weights centred on the median distance
    w = np.exp(-d / max(np.median(d), 1e-9))
    w = w / w.sum() * len(w)
    mi = mutual_info_classif(X, y, random_state=rs.RANDOM_STATE, n_neighbors=3)
    # fold MI into the weighted sample by ranking on mi * mean weight
    score = mi * np.array([w[X[c].notna().values].mean() for c in X.columns])
    order = np.argsort(-score)
    return [X.columns[i] for i in order[:k]]


# --------------------------------------------------------------------------
# AEFS pipeline with selectable stages (used for the full method and ablations)
# --------------------------------------------------------------------------
def run_stages(fold: dict, stages: dict) -> dict:
    """Run the AEFS selection stages requested by ``stages``.

    stages = {"rup": bool, "sfs": bool, "shap": bool, "stability": bool}
    Returns dict(features=..., jaccard=..., k=...).
    """
    X_sm, y_sm = fold["X_train"], fold["y_train"]          # post-RUP + SMOTE
    X_pre, y_pre = fold["X_presmote"], fold["y_presmote"]  # post-RUP, pre-SMOTE
    proj = fold["project_labels"]
    full_cols = list(fold["X_all"].columns)

    if stages.get("rup", True):
        candidates = list(fold["rup_candidates"])
        X_in, y_in = X_sm, y_sm
        X_pre_in = X_pre
    else:
        # Ablation without RUP: work in the complete pre-RUP space, which needs
        # its own SMOTE pass over all columns (fold["X_train"] only carries the
        # RUP-surviving columns).
        candidates = full_cols
        X_in, y_in = rs.apply_smote(fold["X_all"], fold["y_presmote"],
                                    random_state=rs.RANDOM_STATE)
        X_pre_in = fold["X_all"]

    if stages.get("sfs", True):
        candidates = rs.sequential_forward_selection(X_in, y_in, candidates)
    if stages.get("shap", True):
        candidates = rs.shap_refine(X_in, y_in, candidates, 0.02)
    if stages.get("stability", True):
        feats, jac = rs.stability_check(X_pre_in, y_pre, proj, candidates, 0.30)
    else:
        feats, jac = list(candidates), float("nan")
    return dict(features=feats, jaccard=jac, k=len(feats))


def build_method_features(fold: dict, method: str, k_budget: int | None = None) -> dict:
    """Feature set for one method on one fold (selection cost included).

    ``k_budget`` is AEFS's own per-fold feature count.  The k-matched
    baselines need it; passing it in avoids re-running the whole AEFS
    pipeline once per baseline on the same fold.
    """
    X_all, y_pre, X_te_all = fold_fullspace(fold)
    k_aefs = k_budget

    if method == "AEFS":
        out = run_stages(fold, dict(rup=True, sfs=True, shap=True, stability=True))
        return out
    if method == "Ablation_no2D":
        return run_stages(fold, dict(rup=True, sfs=True, shap=True, stability=False))
    if method == "Ablation_noRUP":
        return run_stages(fold, dict(rup=False, sfs=True, shap=True, stability=True))
    if method == "Ablation_noSFS":
        return run_stages(fold, dict(rup=True, sfs=False, shap=True, stability=True))
    if method == "Ablation_noSHAP":
        return run_stages(fold, dict(rup=True, sfs=True, shap=False, stability=True))

    # --- baselines need AEFS's per-fold budget k for a fair comparison ---
    if k_aefs is None:
        k_aefs = run_stages(fold, dict(rup=True, sfs=True, shap=True,
                                       stability=True))["k"]
    if method == "AllFeatures":
        return dict(features=list(X_all.columns), jaccard=float("nan"),
                    k=X_all.shape[1])
    if method == "FilterMI":
        return dict(features=select_filter_mi(X_all, y_pre, k_aefs),
                    jaccard=float("nan"), k=k_aefs)
    if method == "WrapperRFE":
        return dict(features=select_wrapper_rfe(X_all, y_pre, k_aefs),
                    jaccard=float("nan"), k=k_aefs)
    if method == "BorutaSHAP":
        return dict(features=select_borutashap(X_all, y_pre),
                    jaccard=float("nan"), k=None)
    if method == "MI_DS":
        return dict(features=select_mi_ds(X_all, y_pre, k_aefs, X_te_all),
                    jaccard=float("nan"), k=k_aefs)
    if method == "MI_TW":
        return dict(features=select_mi_tw(X_all, y_pre, k_aefs, X_te_all),
                    jaccard=float("nan"), k=k_aefs)
    raise ValueError(f"unknown method {method!r}")


def training_data_for(fold: dict, method: str, feats: list,
                      seed: int | None = None) -> tuple:
    """(X_train, y_train, X_test) with SMOTE applied for every method.

    AEFS and its RUP-based ablations inherit the fold's SMOTE pass (fit on the
    RUP columns); the no-RUP ablation and every baseline re-apply SMOTE on the
    columns they actually use, so no method trains on an unbalanced set.
    """
    if method in ("AEFS", "Ablation_no2D", "Ablation_noSFS", "Ablation_noSHAP"):
        return fold["X_train"], fold["y_train"], fold["X_test"]
    X_all, y_pre, X_te_all = fold_fullspace(fold)
    X_tr, y_tr = rs.apply_smote(X_all[feats], y_pre, random_state=seed)
    return X_tr, y_tr, X_te_all


def run_fold(fold: dict, pool_name: str, methods: list, models: list,
             seed: int | None = None) -> list:
    """Evaluate every (method, model) combination on one LOPO fold."""
    seed = rs.RANDOM_STATE if seed is None else seed
    rows = []
    specs: dict = {}
    # AEFS first: the k-matched baselines reuse its per-fold feature budget
    # instead of re-running the four stages several times on the same fold.
    if any(m in methods for m in ("FilterMI", "WrapperRFE", "MI_DS", "MI_TW")):
        specs["AEFS"] = build_method_features(fold, "AEFS")
    k_budget = specs.get("AEFS", {}).get("k")

    for method in methods:
        if method not in specs:
            specs[method] = build_method_features(fold, method, k_budget=k_budget)
        spec = specs[method]
        feats = spec["features"]
        if not feats:
            continue
        X_tr, y_tr, X_te = training_data_for(fold, method, feats, seed=seed)
        for model in models:
            y_pred, y_proba = fit_predict(model, X_tr, y_tr, X_te, feats)
            row = dict(pool=pool_name, target=fold["target"], method=method,
                       model=model, seed=seed, n_features=len(feats),
                       mean_jaccard=(round(float(spec["jaccard"]), 4)
                                     if np.isfinite(spec["jaccard"]) else np.nan))
            row.update(score_predictions(fold["y_test"], y_pred, y_proba))
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def tasks_for(pools: list) -> list:
    """(pool, target) work items, interleaved so every shard gets every pool."""
    per_pool = []
    for pool in pools:
        data = load_pool(pool)
        for name in data:
            per_pool.append((pool, name))
    # round-robin across pools so a shard never starves
    ordered, buckets = [], {p: [] for p in pools}
    for p, t in per_pool:
        buckets[p].append((p, t))
    while any(buckets.values()):
        for p in pools:
            if buckets[p]:
                ordered.append(buckets[p].pop(0))
    return ordered


def experiment_main(pools: list, shard: int, nshards: int,
                    seed: int | None = None) -> pd.DataFrame:
    """EXP-001: the canonical 6-method x 3-classifier grid."""
    seed = rs.RANDOM_STATE if seed is None else seed
    rows = []
    for pool in pools:
        data = load_pool(pool)
        folds = {f["target"]: f for f in generate_lopo_folds(data)}
        mine = [t for i, t in enumerate(folds) if i % nshards == shard]
        models = models_for(pool)
        print(f"[{pool}] shard {shard}/{nshards}: {len(mine)}/{len(folds)} folds "
              f"models={models} {mine}", flush=True)
        for target in mine:
            rows.extend(run_fold(folds[target], pool, MAIN_METHODS, models,
                                 seed=seed))
    return pd.DataFrame(rows)


def experiment_ablation(pools: list, shard: int, nshards: int,
                        seed: int | None = None) -> pd.DataFrame:
    """EXP-002: component-wise ablation (AEFS and no2D come from EXP-001)."""
    seed = rs.RANDOM_STATE if seed is None else seed
    rows = []
    for pool in pools:
        data = load_pool(pool)
        folds = {f["target"]: f for f in generate_lopo_folds(data)}
        mine = [t for i, t in enumerate(folds) if i % nshards == shard]
        models = models_for(pool)
        print(f"[{pool}] shard {shard}/{nshards}: {len(mine)}/{len(folds)} folds "
              f"models={models}", flush=True)
        for target in mine:
            rows.extend(run_fold(folds[target], pool, ABLATION_METHODS,
                                 models, seed=seed))
    return pd.DataFrame(rows)


def experiment_cpdp(pools: list, shard: int, nshards: int,
                    seed: int | None = None) -> pd.DataFrame:
    """EXP-004: CPDP-aware filter baselines."""
    seed = rs.RANDOM_STATE if seed is None else seed
    rows = []
    for pool in pools:
        data = load_pool(pool)
        folds = {f["target"]: f for f in generate_lopo_folds(data)}
        mine = [t for i, t in enumerate(folds) if i % nshards == shard]
        models = models_for(pool)
        print(f"[{pool}] shard {shard}/{nshards}: {len(mine)}/{len(folds)} folds "
              f"models={models}", flush=True)
        for target in mine:
            rows.extend(run_fold(folds[target], pool, CPDP_METHODS, models,
                                 seed=seed))
    return pd.DataFrame(rows)


def experiment_multiseed(pools: list, shard: int, nshards: int,
                         seeds: list | None = None) -> pd.DataFrame:
    """EXP-003: seed robustness of the full AEFS pipeline."""
    seeds = list(seeds or DEFAULT_SEEDS)
    rows = []
    saved = dict(rs.LGBM_FINAL_PARAMS)
    try:
        for pool in pools:
            data = load_pool(pool)
            for seed in seeds:
                rs.RANDOM_STATE = seed
                rs.LGBM_FINAL_PARAMS["random_state"] = seed
                folds = {f["target"]: f for f in generate_lopo_folds(data)}
                mine = [t for i, t in enumerate(folds) if i % nshards == shard]
                print(f"[{pool}] seed {seed}: {len(mine)}/{len(folds)} folds",
                      flush=True)
                models = models_for(pool)
                for target in mine:
                    rows.extend(run_fold(folds[target], pool, ["AEFS"],
                                         models, seed=seed))
    finally:
        rs.LGBM_FINAL_PARAMS.clear()
        rs.LGBM_FINAL_PARAMS.update(saved)
        rs.RANDOM_STATE = 42
    return pd.DataFrame(rows)


def _sens_row(pool: str, target: str, experiment: str, setting, method: str,
              **kw) -> dict:
    row = dict(pool=pool, target=target, experiment=experiment,
               setting=str(setting), method=method, model="lightgbm",
               seed=rs.RANDOM_STATE, n_features=np.nan,
               mean_jaccard=np.nan, runtime_s=np.nan,
               test_mcc=np.nan, test_f1=np.nan, test_auc_roc=np.nan)
    row.update(kw)
    return row


def _sensitivity_fold(fold: dict, pool: str) -> list:
    """EXP-005 for one LOPO fold: SHAP sweep, stability sweep, convergence and
    runtime - the four experiments ``regen_sensitivity.py`` runs for AEEEM and
    TeraPromise, here for NASA MDP / ReLink with the canonical pipeline."""
    rows = []
    target = fold["target"]
    X_tr, y_tr = fold["X_train"], fold["y_train"]          # post-RUP + SMOTE
    X_te, y_te = fold["X_test"], fold["y_test"]
    X_pre, y_pre = fold["X_presmote"], fold["y_presmote"]  # post-RUP, pre-SMOTE

    # 1) SHAP mass sweep: one SFS pass, then SHAP + stability per threshold
    sfs_feats = rs.sequential_forward_selection(X_tr, y_tr, fold["rup_candidates"])
    for thr in SENS_SHAP_GRID:
        shap_feats = rs.shap_refine(X_tr, y_tr, sfs_feats, thr)
        final, jac = rs.stability_check(X_pre, y_pre, fold["project_labels"],
                                        shap_feats, 0.30)
        m = rs._evaluate(X_tr, y_tr, X_te, y_te, final)
        rows.append(_sens_row(pool, target, "shap", f"{thr:g}", "AEFS",
                              n_features=m["n_features"],
                              mean_jaccard=round(float(jac), 4),
                              test_mcc=m["mcc"], test_f1=m["f1"],
                              test_auc_roc=m["auc"]))

    # 2) stability sweep: SFS + SHAP once, then stability per threshold
    shap_feats = rs.shap_refine(X_tr, y_tr, sfs_feats, 0.02)
    for thr in SENS_STABILITY_GRID:
        final, jac = rs.stability_check(X_pre, y_pre, fold["project_labels"],
                                        shap_feats, thr)
        m = rs._evaluate(X_tr, y_tr, X_te, y_te, final)
        rows.append(_sens_row(pool, target, "stability", f"{thr:g}", "AEFS",
                              n_features=m["n_features"],
                              mean_jaccard=round(float(jac), 4),
                              test_mcc=m["mcc"], test_f1=m["f1"],
                              test_auc_roc=m["auc"]))

    # 3) + 4) one instrumented full pipeline: stage timings (runtime rows) and
    #    the feature count after every stage (convergence rows)
    rows.append(_sens_row(pool, target, "convergence", "original", "AEFS",
                          n_features=fold["X_all"].shape[1]))
    t0 = time.perf_counter()
    candidates = rs.compute_rup_features(fold["X_all"], y_pre)
    secs_rup = time.perf_counter() - t0
    rows.append(_sens_row(pool, target, "convergence", "after_rup", "AEFS",
                          n_features=len(candidates)))

    t0 = time.perf_counter()
    sfs2 = rs.sequential_forward_selection(X_tr, y_tr, candidates)
    secs_sfs = time.perf_counter() - t0
    rows.append(_sens_row(pool, target, "convergence", "after_sfs", "AEFS",
                          n_features=len(sfs2)))

    t0 = time.perf_counter()
    shap2 = rs.shap_refine(X_tr, y_tr, sfs2, 0.02)
    secs_shap = time.perf_counter() - t0
    rows.append(_sens_row(pool, target, "convergence", "after_shap", "AEFS",
                          n_features=len(shap2)))

    t0 = time.perf_counter()
    final, jac = rs.stability_check(X_pre, y_pre, fold["project_labels"],
                                    shap2, 0.30)
    secs_stab = time.perf_counter() - t0
    rows.append(_sens_row(pool, target, "convergence", "after_stability",
                          "AEFS", n_features=len(final),
                          mean_jaccard=round(float(jac), 4)))

    t0 = time.perf_counter()
    lgbm = LGBMClassifier(**rs.LGBM_FINAL_PARAMS)
    lgbm.fit(X_tr[final], y_tr)
    lgbm.predict(X_te[final])
    secs_fit = time.perf_counter() - t0

    for setting, secs in (("rup", secs_rup), ("sfs", secs_sfs),
                          ("shap", secs_shap), ("stability", secs_stab),
                          ("fit_predict", secs_fit)):
        rows.append(_sens_row(pool, target, "runtime", f"AEFS.{setting}",
                              "AEFS", runtime_s=round(secs, 3)))
    rows.append(_sens_row(pool, target, "runtime", "AEFS.total", "AEFS",
                          runtime_s=round(secs_rup + secs_sfs + secs_shap
                                          + secs_stab + secs_fit, 3),
                          n_features=len(final)))

    # baselines: selection + fit + predict on the pre-RUP space (SMOTE applied
    # there too), timed exactly like regen_sensitivity.measure_runtime
    X_all_sm, y_all_sm = rs.apply_smote(fold["X_all"], y_pre)
    X_te_all = fold["X_te_all"]
    k = len(final)

    t0 = time.perf_counter()
    LGBMClassifier(**rs.LGBM_FINAL_PARAMS).fit(X_all_sm, y_all_sm).predict(X_te_all)
    rows.append(_sens_row(pool, target, "runtime", "AllFeatures.pipeline",
                          "AllFeatures", runtime_s=round(time.perf_counter() - t0, 3),
                          n_features=X_all_sm.shape[1]))

    t0 = time.perf_counter()
    sel = SelectKBest(score_func=lambda X_, y_: mutual_info_classif(
        X_, y_, random_state=rs.RANDOM_STATE), k=min(k, X_all_sm.shape[1]))
    sel.fit(X_all_sm, y_all_sm)
    feats = list(X_all_sm.columns[sel.get_support()])
    LGBMClassifier(**rs.LGBM_FINAL_PARAMS).fit(X_all_sm[feats], y_all_sm).predict(X_te_all[feats])
    rows.append(_sens_row(pool, target, "runtime", "FilterMI.pipeline",
                          "FilterMI", runtime_s=round(time.perf_counter() - t0, 3),
                          n_features=len(feats)))

    t0 = time.perf_counter()
    rfe = RFE(LGBMClassifier(**RFE_ESTIMATOR_PARAMS, random_state=rs.RANDOM_STATE),
              n_features_to_select=min(k, X_all_sm.shape[1]), step=0.2)
    rfe.fit(X_all_sm, y_all_sm)
    feats = list(X_all_sm.columns[rfe.support_])
    LGBMClassifier(**rs.LGBM_FINAL_PARAMS).fit(X_all_sm[feats], y_all_sm).predict(X_te_all[feats])
    rows.append(_sens_row(pool, target, "runtime", "WrapperRFE.pipeline",
                          "WrapperRFE", runtime_s=round(time.perf_counter() - t0, 3),
                          n_features=len(feats)))

    return rows


def experiment_sensitivity(pools: list, shard: int, nshards: int) -> pd.DataFrame:
    """EXP-005: sensitivity / convergence / runtime for the pools that never
    got them (audit A10).  Sharded per fold, like the other experiments."""
    rows = []
    for pool in pools:
        folds = generate_lopo_folds(load_pool(pool))
        mine = [i for i in range(len(folds)) if i % nshards == shard]
        print(f"[{pool}] sensitivity shard {shard}/{nshards}: "
              f"{len(mine)}/{len(folds)} folds "
              f"{[folds[i]['target'] for i in mine]}", flush=True)
        for i in mine:
            rows.extend(_sensitivity_fold(folds[i], pool))
    return pd.DataFrame(rows, columns=SENS_COLUMNS)


EXPERIMENTS = {
    "main": experiment_main,
    "ablation": experiment_ablation,
    "cpdp": experiment_cpdp,
    "multiseed": experiment_multiseed,
    "sensitivity": experiment_sensitivity,
}
