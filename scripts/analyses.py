"""Statistics for the manuscript (ANL-001 .. ANL-005 in docs/EXPERIMENTS.md).

Everything here reads the *verified* experiment outputs - never the legacy
tables - so a number in the paper always traces back to
``results/experiments/<job>/all_rows.csv``.

    ANL-001  descriptive.csv   bootstrap 95% CIs for MCC / F1 / AUC / #features
    ANL-002  rq1_rq2.csv       Wilcoxon + Holm + Cliff's delta, Friedman + Nemenyi
    ANL-003  rq3.csv           paired ablation effect sizes (EXP-002 + no2D vs AEFS)
    ANL-004  rq4.csv           critical-difference ranks for TeraPromise
    ANL-005  rq5.csv           defect rate / source-target similarity vs AEFS MCC

Usage:
    python scripts/analyses.py                 # run every analysis
    python scripts/analyses.py --only anl001
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aefs_core as core  # noqa: E402

EXP = ROOT / "results" / "experiments"
OUT = ROOT / "results" / "analysis"
BOOTSTRAP = 5000
SEED = 42
METHOD_ORDER = ["AEFS", "Ablation_no2D", "Ablation_noRUP", "Ablation_noSFS",
                "Ablation_noSHAP", "AllFeatures", "FilterMI", "WrapperRFE",
                "BorutaSHAP"]
# Nemenyi critical values for alpha = 0.05 (Demšar 2006, Table 5)
NEMENYI_Q = {2: 2.728, 3: 3.314, 4: 3.633, 5: 3.858, 6: 4.030}


def load(job: str) -> pd.DataFrame:
    path = EXP / job / "all_rows.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return pd.read_csv(path)


def bootstrap_ci(x: np.ndarray, stat=np.mean, b: int = BOOTSTRAP,
                 seed: int = SEED) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0]), float(x[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(b, len(x)))
    draws = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def cliffs_delta(a, b) -> float:
    """P(a > b) - P(a < b); 0 when either sample is empty."""
    a = np.asarray([v for v in a if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in b if np.isfinite(v)], dtype=float)
    if not len(a) or not len(b):
        return np.nan
    diff = (a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()
    return float(diff / (len(a) * len(b)))


def holm(p: dict) -> dict:
    items = sorted(((k, v) for k, v in p.items() if np.isfinite(v)),
                   key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for rank, (k, v) in enumerate(items):
        running = max(running, (m - rank) * v)
        out[k] = min(1.0, running)
    return out


# --------------------------------------------------------------------------
# ANL-001
# --------------------------------------------------------------------------
def anl001() -> pd.DataFrame:
    df = load("main")
    rows = []
    for (pool, method, model), g in df.groupby(["pool", "method", "model"]):
        rec = dict(pool=pool, method=method, model=model, n=len(g))
        for col, prefix in (("test_mcc", "mcc"), ("test_f1", "f1"),
                            ("test_auc_roc", "auc"), ("n_features", "nfeat")):
            m, lo, hi = bootstrap_ci(g[col].to_numpy())
            rec[f"{prefix}_mean"] = round(m, 4)
            rec[f"{prefix}_lo"] = round(lo, 4)
            rec[f"{prefix}_hi"] = round(hi, 4)
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["pool", "method", "model"])
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "descriptive.csv", index=False, lineterminator="\n")
    print(f"[anl001] {len(out)} rows -> analysis/descriptive.csv")
    return out


# --------------------------------------------------------------------------
# ANL-002
# --------------------------------------------------------------------------
def anl002() -> pd.DataFrame:
    df = load("main")
    # the legacy-schema mirrors keep the comparison sets identical to the paper
    baselines = ["Ablation_no2D", "AllFeatures", "FilterMI", "WrapperRFE",
                 "BorutaSHAP"]
    rows = []
    for (pool, model), g in df.groupby(["pool", "model"]):
        wide = g.pivot_table(index="target", columns="method",
                             values="test_mcc", aggfunc="first")
        if "AEFS" not in wide:
            continue
        pvals, made = {}, 0
        for base in baselines:
            if base not in wide:
                continue
            pair = wide[["AEFS", base]].dropna()
            if len(pair) < 3:
                continue
            try:
                stat, p = stats.wilcoxon(pair["AEFS"], pair[base])
            except ValueError:
                p = 1.0
            # scipy returns NaN when every paired difference is zero; such a
            # comparison has no evidence either way, so it is reported but
            # kept out of the Holm family (see stats_q1.py: same rule).
            if np.isfinite(p):
                pvals[base] = p
            d = cliffs_delta(pair["AEFS"], pair[base])
            rows.append(dict(pool=pool, model=model, test="wilcoxon",
                             comparison=f"AEFS vs {base}", n=len(pair),
                             statistic=float(stat), p_value=float(p),
                             effect_size=d, effect_name="cliffs_delta",
                             mean_diff=float(pair["AEFS"].mean() - pair[base].mean())))
            made += 1
        adj = holm(pvals)
        for r in rows[-made:]:
            base = r["comparison"].replace("AEFS vs ", "")
            r["p_holm"] = adj.get(base, np.nan)
        # Friedman over the methods that are present in every target
        methods = [m for m in METHOD_ORDER if m in wide]
        complete = wide[methods].dropna()
        if len(complete) >= 3 and len(methods) >= 3:
            try:
                chi2, p_f = stats.friedmanchisquare(*[complete[m] for m in methods])
            except ValueError:
                chi2, p_f = np.nan, np.nan
            ranks = complete.rank(axis=1, ascending=False).mean()
            n = len(complete)
            k = len(methods)
            cd = NEMENYI_Q.get(k, 0) * np.sqrt(k * (k + 1) / (6 * n)) if k in NEMENYI_Q else np.nan
            for m in methods:
                rows.append(dict(pool=pool, model=model, test="friedman",
                                 comparison=f"all {k} methods", n=n,
                                 statistic=float(chi2), p_value=float(p_f),
                                 effect_size=float(ranks[m]),
                                 effect_name="avg_rank", p_holm=np.nan,
                                 mean_diff=np.nan, nemenyi_cd=float(cd)))
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "rq1_rq2.csv", index=False, lineterminator="\n")
    print(f"[anl002] {len(out)} rows -> analysis/rq1_rq2.csv")
    return out


# --------------------------------------------------------------------------
# ANL-003
# --------------------------------------------------------------------------
def anl003() -> pd.DataFrame:
    main = load("main")
    abl = load("ablation")
    variants = ["Ablation_no2D", "Ablation_noRUP", "Ablation_noSFS",
                "Ablation_noSHAP"]
    df = pd.concat([main[main.method.isin(["AEFS"] + variants)],
                    abl], ignore_index=True)
    rows = []
    for (pool, model, variant), g in df.groupby(["pool", "model", "method"]):
        if variant == "AEFS":
            continue
        base = df[(df.pool == pool) & (df.model == model)
                  & (df.method == "AEFS")].set_index("target")
        cur = g.set_index("target")
        pair = cur.join(base[["test_mcc", "n_features"]],
                        rsuffix="_aefs").dropna(subset=["test_mcc", "test_mcc_aefs"])
        if len(pair) < 3:
            continue
        diff = pair["test_mcc"] - pair["test_mcc_aefs"]
        _, lo, hi = bootstrap_ci(diff.to_numpy())
        try:
            _, p = stats.wilcoxon(pair["test_mcc"], pair["test_mcc_aefs"])
        except ValueError:
            p = 1.0
        rows.append(dict(pool=pool, model=model, variant=variant, n=len(pair),
                         aefs_mcc=round(float(pair["test_mcc_aefs"].mean()), 4),
                         variant_mcc=round(float(pair["test_mcc"].mean()), 4),
                         delta_mcc=round(float(diff.mean()), 4),
                         delta_lo=round(lo, 4), delta_hi=round(hi, 4),
                         cliffs_delta=round(cliffs_delta(pair["test_mcc"],
                                                         pair["test_mcc_aefs"]), 4),
                         p_value=float(p),
                         aefs_nfeat=round(float(pair["n_features_aefs"].mean()), 3),
                         variant_nfeat=round(float(pair["n_features"].mean()), 3)))
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "rq3.csv", index=False, lineterminator="\n")
    print(f"[anl003] {len(out)} rows -> analysis/rq3.csv")
    return out


# --------------------------------------------------------------------------
# ANL-004
# --------------------------------------------------------------------------
def anl004() -> pd.DataFrame:
    df = load("main")
    rows = []
    for pool in sorted(df.pool.unique()):
        for model in sorted(df[df.pool == pool].model.unique()):
            g = df[(df.pool == pool) & (df.model == model)]
            wide = g.pivot_table(index="target", columns="method",
                                 values="test_mcc", aggfunc="first").dropna()
            methods = [m for m in METHOD_ORDER if m in wide]
            if len(methods) < 3 or len(wide) < 3:
                continue
            chi2, p = stats.friedmanchisquare(*[wide[m] for m in methods])
            ranks = wide[methods].rank(axis=1, ascending=False).mean()
            n, k = len(wide), len(methods)
            cd = (NEMENYI_Q.get(k, np.nan)
                  * np.sqrt(k * (k + 1) / (6 * n)) if k in NEMENYI_Q else np.nan)
            for m in methods:
                rows.append(dict(pool=pool, model=model, method=m,
                                 avg_rank=round(float(ranks[m]), 4),
                                 n_targets=n, k_methods=k,
                                 friedman_chi2=round(float(chi2), 4),
                                 friedman_p=float(p),
                                 nemenyi_cd=round(float(cd), 4)))
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "rq4.csv", index=False, lineterminator="\n")
    print(f"[anl004] {len(out)} rows -> analysis/rq4.csv")
    return out


# --------------------------------------------------------------------------
# ANL-005
# --------------------------------------------------------------------------
def _target_characteristics(pool: str, target: str, max_sources: int = 15) -> dict:
    data = core.load_pool(pool)
    if target not in data:
        return {}
    tgt = data[target]
    src = [d for name, d in data.items() if name != target]
    rng = np.random.default_rng(SEED)
    if len(src) > max_sources:
        pick = rng.choice(len(src), size=max_sources, replace=False)
        src = [src[i] for i in sorted(pick)]
    sims = []
    for s in src:
        feats = [c for c in tgt.columns if c != "defective" and c in s.columns]
        sims.append(float(np.mean([
            core._distribution_similarity(s[c], tgt[c]) for c in feats])))
    return dict(defect_rate=float(tgt["defective"].mean()),
                n_rows=int(len(tgt)),
                n_features=int(tgt.shape[1] - 1),
                sim_sources=len(src),
                src_target_similarity=float(np.mean(sims)) if sims else np.nan)


def anl005() -> pd.DataFrame:
    main = load("main")
    aefs = (main[main.method == "AEFS"].groupby(["pool", "target"])
            .agg(mcc=("test_mcc", "mean"), n_features=("n_features", "mean"))
            .reset_index())
    rows = []
    for pool in sorted(aefs.pool.unique()):
        data = core.load_pool(pool)
        for target in sorted(aefs[aefs.pool == pool].target.unique()):
            if target not in data:
                continue
            ch = _target_characteristics(pool, target)
            if not ch:
                continue
            rec = aefs[(aefs.pool == pool) & (aefs.target == target)].iloc[0]
            rows.append(dict(pool=pool, target=target,
                             defect_rate=round(ch["defect_rate"], 4),
                             n_rows=ch["n_rows"], n_features=ch["n_features"],
                             src_target_similarity=round(ch["src_target_similarity"], 4),
                             sim_sources=ch["sim_sources"],
                             aefs_mcc=round(float(rec["mcc"]), 4),
                             aefs_n_features=int(round(rec["n_features"]))))
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "rq5_raw.csv", index=False, lineterminator="\n")
    # correlation summary (pooled and per pool)
    summary = []
    for scope, g in [("ALL", out)] + [(p, out[out.pool == p])
                                      for p in sorted(out.pool.unique())]:
        for x in ["defect_rate", "src_target_similarity"]:
            sub = g[[x, "aefs_mcc"]].dropna()
            if len(sub) < 4:
                continue
            r_s, p_s = stats.spearmanr(sub[x], sub["aefs_mcc"])
            r_p, p_p = stats.pearsonr(sub[x], sub["aefs_mcc"])
            summary.append(dict(scope=scope, predictor=x, n=len(sub),
                                spearman_r=round(float(r_s), 4),
                                spearman_p=float(p_s),
                                pearson_r=round(float(r_p), 4),
                                pearson_p=float(p_p)))
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "rq5.csv", index=False, lineterminator="\n")
    print(f"[anl005] {len(out)} targets -> analysis/rq5_raw.csv, "
          f"{len(summ)} correlations -> analysis/rq5.csv")
    return summ


# --------------------------------------------------------------------------
# ANL-006 - every number printed in the three manuscripts vs results/
# --------------------------------------------------------------------------
TEX_FILES = [ROOT / "paper" / "main.tex", ROOT / "paper" / "main_ist.tex",
             ROOT / "paper-ist" / "main.tex"]
METHOD_LABEL = {"AEFS": "AEFS", "Ablation": "Ablation_no2D",
                "Ablation_no2D": "Ablation_no2D", "All": "AllFeatures",
                "AllFeatures": "AllFeatures", "FilterMI": "FilterMI",
                "WrapperRFE": "WrapperRFE", "BorutaSHAP": "BorutaSHAP",
                "Boruta": "BorutaSHAP", "MI-DS": "MI_DS", "MI-TW": "MI_TW"}
MODEL_LABEL = {"LGBM": "lightgbm", "RF": "random_forest", "SVM": "svm"}
POOL_LABEL = {"AEEEM": "aeeem", "TeraPromise": "tera", "NASA MDP": "nasa",
              "NASA": "nasa", "ReLink": "relink"}
NUM = r"(-?\d*\.?\d+)"


def _means(csv_path: Path) -> pd.DataFrame:
    """pool x method x model means (and stds) from a results table."""
    df = pd.read_csv(csv_path)
    g = df.groupby(["pool", "method", "model"])
    out = g.agg(mcc=("test_mcc", "mean"), mcc_sd=("test_mcc", "std"),
                f1=("test_f1", "mean"), auc=("test_auc_roc", "mean"),
                nfeat=("n_features", "mean"), nfeat_sd=("n_features", "std"),
                jac=("mean_jaccard", "mean")).reset_index()
    return out


def _grid(path: Path) -> dict:
    """{(pool, method, model): {metric: value}} for fast lookup."""
    if not path.exists():
        return {}
    df = _means(path)
    return {(r.pool, r.method, r.model): r._asdict() for r in df.itertuples()}


def _tol(raw: str) -> float:
    """Compare at the precision the manuscript actually prints."""
    dec = len(raw.split(".")[1]) if "." in raw else 0
    return 0.5 * 10 ** (-dec) + 1e-9


def _num(cell: str) -> tuple | None:
    """(value, tolerance) for the first number in a table cell."""
    m = re.search(NUM, cell.replace(",", ""))
    if not m:
        return None
    raw = m.group(1)
    try:
        v = float(raw)
    except ValueError:
        return None
    return v, _tol(raw)


def _match(value: tuple | None, grid: dict, key: tuple, metric: str,
           tol: float | None = None) -> tuple[str, float | None]:
    if value is None or not grid:
        return "NO_TEX_VALUE", None
    if tol is None:
        tol = value[1]
    value = value[0]
    rec = grid.get(key)
    if rec is None or rec.get(metric) is None or not np.isfinite(rec[metric]):
        return "NO_REFERENCE", None
    exp = float(rec[metric])
    return ("MATCH" if abs(value - exp) <= tol else "MISMATCH"), exp


def anl006() -> pd.DataFrame:
    legacy = {p: _grid(ROOT / "results" / f"{p}_stage4_full_results.csv")
              for p in POOL_LABEL.values()}
    new = {p: _grid(EXP / "main" / f"{p}_results.csv")
           for p in POOL_LABEL.values()}
    # EXP-004 CPDP-specific baselines (MI-DS/MI-TW) live in their own grid;
    # table cells for those rows fall back to it when main/ has no such key
    cpdp = {p: _grid(EXP / "cpdp" / f"{p}_results.csv")
            for p in POOL_LABEL.values()}
    rows = []
    for tex in TEX_FILES:
        if not tex.exists():
            continue
        lines = tex.read_text(encoding="utf-8", errors="replace").splitlines()
        pool, method, model, in_tab = None, None, None, None
        for i, raw in enumerate(lines, 1):
            line = raw.strip()
            if line.startswith("\\begin{table"):
                pool, method, model, in_tab = None, None, None, None
            m = re.search(r"\\label\{tab:(\w+)\}", line)
            if m:
                in_tab = m.group(1)
                if in_tab.endswith("_results"):
                    pool = in_tab[:-len("_results")]
            mm = re.search(r"\\multirow.*textbf\{(LGBM|RF|SVM)\}", line)
            if mm:
                model = MODEL_LABEL[mm.group(1)]
            if in_tab == "datasets" and "&" in line:
                cells = [c.strip() for c in line.split("&")]
                if cells and cells[0] in POOL_LABEL:
                    rows.append(_dataset_claim(tex, i, POOL_LABEL[cells[0]],
                                               cells))
                continue
            if pool and "&" in line:
                cells = [c.strip() for c in line.split("&")]
                for c in cells:
                    mm = re.search(r"textbf\{(LGBM|RF|SVM)\}", c)
                    if mm:
                        model = MODEL_LABEL[mm.group(1)]
                # the method cell is not always the first one (multirow model)
                idx = None
                for j, c in enumerate(cells):
                    lab = re.sub(r"\\textbf\{|\}", "", c).strip()
                    if lab in METHOD_LABEL:
                        idx, lab = j, lab
                        break
                if idx is not None and len(cells) - idx >= 6:
                    method = METHOD_LABEL[lab]
                    nums = [_num(c.replace("\\", " ")) for c in cells[idx + 1:]]
                    # cells: method, |F*|, J, MCC, F1, AUC  (mean[+-]std)
                    for metric, pos in (("nfeat", 0), ("jac", 1), ("mcc", 2),
                                        ("f1", 3), ("auc", 4)):
                        if pos >= len(nums) or nums[pos] is None:
                            continue
                        item = nums[pos]
                        key = (pool, method, model)
                        grid = new[pool] if key in new[pool] else cpdp[pool]
                        st_new, exp_new = _match(item, grid, key, metric)
                        st_old, exp_old = _match(item, legacy[pool], key,
                                                 metric)
                        rows.append(dict(file=tex.name, line=i,
                                         claim=f"tab:{pool} {model}/{method}/{metric}",
                                         value_tex=item[0],
                                         value_legacy=exp_old,
                                         value_new=exp_new,
                                         status=_joint(st_old, st_new)))
                continue
            # ---- prose claims -------------------------------------------------
            rows.extend(_prose_claims(tex, i, line, legacy, new))

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "rq6.csv", index=False, lineterminator="\n")
    counts = out.status.value_counts().to_dict()
    print(f"[anl006] {len(out)} claims -> analysis/rq6.csv {counts}")
    return out


def _joint(st_old: str, st_new: str) -> str:
    if "NO_TEX_VALUE" in (st_old, st_new):
        return "NO_TEX_VALUE"
    if st_old == "MATCH" and st_new == "MATCH":
        return "MATCH_BOTH"
    if st_old == "MATCH":
        return "MATCH_LEGACY"
    if st_new == "MATCH":
        return "MATCH_NEW"
    if "MISMATCH" in (st_old, st_new):
        return "MISMATCH"
    return "NO_REFERENCE"


_POOL_CACHE: dict = {}


def _pool(pool: str) -> dict:
    if pool not in _POOL_CACHE:
        _POOL_CACHE[pool] = core.load_pool(pool)
    return _POOL_CACHE[pool]


def _dataset_claim(tex: Path, line: int, pool: str, cells: list) -> dict:
    """Check tab:datasets against the loaded pools."""
    data = _pool(pool)
    n_proj = len(data)
    n_feat = next(iter(data.values())).shape[1] - 1
    avg_rows = float(np.mean([len(d) for d in data.values()]))
    rate_lo = min(d["defective"].mean() for d in data.values()) * 100
    rate_hi = max(d["defective"].mean() for d in data.values()) * 100
    def num(pos: int):
        item = _num(cells[pos]) if len(cells) > pos else None
        return item[0] if item else None

    tex_proj = num(1)
    tex_feat = num(2)
    tex_avg = num(3)
    probs = []
    if tex_proj != n_proj:
        probs.append(f"projects tex={tex_proj} data={n_proj}")
    if tex_feat != n_feat:
        probs.append(f"features tex={tex_feat} data={n_feat}")
    if tex_avg and abs(tex_avg - avg_rows) / max(avg_rows, 1) > 0.05:
        probs.append(f"avg_instances tex={tex_avg:.0f} data={avg_rows:.0f}")
    rate = _re_rate(cells[4] if len(cells) > 4 else "")
    if rate and (abs(rate[0] - rate_lo) > 2 or abs(rate[1] - rate_hi) > 2):
        probs.append(f"defect_rate tex={rate} data=({rate_lo:.0f},{rate_hi:.0f})")
    return dict(file=tex.name, line=line, claim=f"tab:datasets {pool}",
                value_tex=tex_proj, value_legacy=None, value_new=None,
                status="MATCH_DATA" if not probs else "DATA_MISMATCH",
                note="; ".join(probs))


def _re_rate(text: str) -> tuple[float, float] | None:
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if len(nums) >= 2:
        return nums[0], nums[1]
    if len(nums) == 1:
        return nums[0], nums[0]
    return None


OUT_OF_SCOPE_WORDS = ("convergence", "Convergence", "threshold", "sensitivity",
                      "Sensitivity", "runtime", "MBL-CPDP", "PROMISE",
                      "extract", "reported in", "Shadow")


def _reduction(pool: str, grid: dict) -> float | None:
    """% of features AEFS removes relative to AllFeatures for one pool."""
    if not grid:
        return None
    all_f = [v["nfeat"] for (p, m, mod), v in grid.items()
             if p == pool and m == "AllFeatures"]
    aefs = [v["nfeat"] for (p, m, mod), v in grid.items()
            if p == pool and m == "AEFS"]
    if not all_f or not aefs:
        return None
    a, e = float(np.mean(all_f)), float(np.mean(aefs))
    return 100.0 * (a - e) / a if a else None


def _delta_bound(sentence: str, methods: list, pools: list,
                 legacy: dict, new: dict) -> tuple[float, float, str] | None:
    """|Delta MCC| <= X / "MCC within X of" claims: largest |AEFS - baseline|."""
    m = re.search(r"\\leq\s*" + NUM, sentence) \
        or re.search(r"within\s+" + NUM + r"\s+of", sentence)
    if not m:
        return None
    value = float(m.group(1))
    if "full-feature" in sentence or "AllFeatures" in sentence:
        bases = ["AllFeatures"]
    else:
        bases = ["BorutaSHAP", "AllFeatures", "FilterMI", "WrapperRFE",
                 "Ablation_no2D"]
    bases = [b for b in methods if b != "AEFS"] or bases
    if not bases or bases == ["AEFS"]:
        bases = ["AllFeatures"]
    worst: dict[str, float | None] = {"legacy": None, "new": None}
    for tag, grid in (("legacy", legacy), ("new", new)):
        for pool, g in grid.items():
            if pools and pool not in pools:
                continue
            for (p, meth, mod), rec in g.items():
                if meth not in bases:
                    continue
                aefs = g.get((p, "AEFS", mod))
                if aefs is None or rec.get("mcc") is None:
                    continue
                d = abs(float(rec["mcc"]) - float(aefs["mcc"]))
                worst[tag] = d if worst[tag] is None else max(worst[tag], d)
    hit = {t for t, v in worst.items() if v is not None and v <= value + 1e-9}
    have = {t for t, v in worst.items() if v is not None}
    st = ("MATCH_BOTH" if hit == have and have else
          "MATCH_LEGACY" if "legacy" in hit else
          "MATCH_NEW" if "new" in hit else "UNMATCHED_BOUND")
    shown = worst["new"] if worst["new"] is not None else worst["legacy"]
    return value, round(shown, 3) if shown is not None else None, st


def _reduction_range(sentence: str, legacy: dict, new: dict):
    """'reduces features by 62--89%' -> per-pool reduction range."""
    m = re.search(r"(\d+)\s*--\s*(\d+)\\?\s*%", sentence)
    if not m or "reduc" not in sentence.lower():
        return None
    lo_t, hi_t = float(m.group(1)), float(m.group(2))
    hit = {}
    out = {}
    for tag, grid in (("legacy", legacy), ("new", new)):
        vals = [v for v in (_reduction(p, grid[p]) for p in grid)
                if v is not None]
        if not vals:
            continue
        lo_c, hi_c = round(min(vals), 1), round(max(vals), 1)
        out[tag] = f"{lo_c}-{hi_c}"
        if abs(lo_c - lo_t) <= 1.5 and abs(hi_c - hi_t) <= 1.5:
            hit[tag] = True
    st = ("MATCH_BOTH" if set(hit) == {"legacy", "new"} else
          "MATCH_LEGACY" if hit.get("legacy") else
          "MATCH_NEW" if hit.get("new") else "UNMATCHED")
    return lo_t, out.get("legacy"), out.get("new"), st


def _prose_claims(tex: Path, line: int, text: str, legacy: dict,
                  new: dict) -> list:
    """Inline "MCC of 0.289" / "with 6.6 features" sentences vs the grids."""
    out = []
    if not any(k in text for k in ("MCC", "features", "F1", "AUC")):
        return out
    sentence = text.replace("$-$", "-").replace("$+$", "+").replace("$", "")
    methods = [m for m in ("AEFS", "FilterMI", "AllFeatures", "WrapperRFE",
                           "BorutaSHAP", "Ablation") if m in sentence]
    pools = [POOL_LABEL[p] for p in POOL_LABEL if p in sentence]
    scope = f"{', '.join(methods) or 'any'}"

    if any(w in sentence for w in OUT_OF_SCOPE_WORDS):
        return [dict(file=tex.name, line=line, claim=f"prose MCC ({scope})",
                     value_tex=None, value_legacy=None, value_new=None,
                     status="OUT_OF_SCOPE",
                     note="sensitivity/convergence/runtime or external number")]

    # "|Delta MCC| <= 0.071" / "MCC within 0.071 of X" - bound on a difference
    if "\\leq" in sentence or re.search(r"within\s+" + NUM + r"\s+of", sentence):
        res = _delta_bound(sentence, methods, pools, legacy, new)
        if res is not None:
            value, exp, st = res
            out.append(dict(file=tex.name, line=line,
                            claim=f"prose |dMCC| bound ({scope})",
                            value_tex=value, value_legacy=None, value_new=exp,
                            status=st))

    # "by 62--89\% across four benchmark datasets" -> reduction range
    if "--" in sentence:
        rng = _reduction_range(sentence, legacy, new)
        if rng is not None:
            lo_t, lo_c, hi_c, st = rng
            out.append(dict(file=tex.name, line=line,
                            claim="prose reduction% range",
                            value_tex=lo_t, value_legacy=lo_c, value_new=hi_c,
                            status=st))
            return out

    # "61.5\% fewer features" -> reduction of AllFeatures -> AEFS
    if "fewer features" in sentence:
        m = re.search(NUM + r"\\?\s*%", sentence)
        if m and pools:
            value = float(m.group(1))
            old = _reduction(pools[0], legacy[pools[0]])
            new_ = _reduction(pools[0], new[pools[0]])
            tol = 0.6  # one decimal place of a percentage
            if old is not None and abs(value - old) <= tol:
                st = "MATCH_BOTH" if new_ is not None and abs(value - new_) <= tol \
                    else "MATCH_LEGACY"
            elif new_ is not None and abs(value - new_) <= tol:
                st = "MATCH_NEW"
            else:
                st = "UNMATCHED"
            out.append(dict(file=tex.name, line=line,
                            claim=f"prose reduction% {pools[0]}",
                            value_tex=value,
                            value_legacy=round(old, 2) if old is not None else None,
                            value_new=round(new_, 2) if new_ is not None else None,
                            status=st))
            return out

    # MCC levels are printed with a decimal point; an integer match here would
    # just be the "1" of "F1" or the "4" of "LOPO"
    for m in re.finditer(r"MCC[^.\d\-]{0,12}(-?\d*\.\d+)", sentence):
        raw = m.group(1)
        if "within" in sentence[m.start():m.end()]:
            continue  # "MCC within 0.071 of ..." handled as a bound above
        value = float(raw)
        if not -0.5 <= value <= 1.0:
            continue
        exp_old, exp_new, st = _prose_close(value, "mcc", methods, pools,
                                            legacy, new, tol=_tol(raw))
        out.append(dict(file=tex.name, line=line,
                        claim=f"prose MCC ({scope})",
                        value_tex=value, value_legacy=exp_old,
                        value_new=exp_new, status=st))
    for m in re.finditer(NUM + r"(?!\s*\\?\s*%)\s*(?:fewer |less )?"
                         r"(?:\d+\s*)?features", sentence):
        raw = m.group(1)
        value = float(raw)
        if value <= 0 or value > 200:
            continue
        exp_old, exp_new, st = _prose_close(value, "nfeat", methods, pools,
                                            legacy, new, tol=_tol(raw))
        out.append(dict(file=tex.name, line=line,
                        claim=f"prose n_features ({scope})",
                        value_tex=value, value_legacy=exp_old,
                        value_new=exp_new, status=st))
    return out


def _prose_close(value: float, metric: str, methods: list, pools: list,
                 legacy: dict, new: dict, tol: float = 0.006):
    best_old = best_new = None
    d_old = d_new = 1e9
    for pool, grid in legacy.items():
        if pools and pool not in pools:
            continue
        for (p, meth, mod), rec in grid.items():
            if methods and meth not in methods:
                continue
            v = rec.get(metric)
            if v is None or not np.isfinite(v):
                continue
            if abs(v - value) < d_old:
                d_old, best_old = abs(v - value), float(v)
    for pool, grid in new.items():
        if pools and pool not in pools:
            continue
        for (p, meth, mod), rec in grid.items():
            if methods and meth not in methods:
                continue
            v = rec.get(metric)
            if v is None or not np.isfinite(v):
                continue
            if abs(v - value) < d_new:
                d_new, best_new = abs(v - value), float(v)
    if best_old is not None and d_old <= tol:
        st = ("MATCH_BOTH" if best_new is not None and d_new <= tol
              else "MATCH_LEGACY")
        return best_old, best_new, st
    if best_new is not None and d_new <= tol:
        return best_old, best_new, "MATCH_NEW"
    return best_old, best_new, "UNMATCHED"


ANALYSES = {"anl001": anl001, "anl002": anl002, "anl003": anl003,
            "anl004": anl004, "anl005": anl005, "anl006": anl006}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(ANALYSES), default=None)
    args = ap.parse_args()
    for name, fn in ANALYSES.items():
        if args.only and name != args.only:
            continue
        fn()


if __name__ == "__main__":
    main()
