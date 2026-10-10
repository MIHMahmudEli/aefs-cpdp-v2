"""Turn a merged experiment output into the tables the rest of the pipeline reads.

Inputs
------
    results/experiments/<job>/all_rows.csv        (written by fanout.py merge)
    results/<pool>_stage4_full_results.csv        (the legacy tables, for audit)

Outputs
-------
    results/experiments/<job>/legacy/<pool>_stage4_full_results.csv
        a drop-in mirror of the legacy schema (the 5 methods x 2 classifiers
        that the legacy tables contain) so ``compute_stats.py``,
        ``fix_results_text.py`` and ``stats_q1.py`` can be pointed at the new
        numbers without being rewritten;
    results/experiments/<job>/coverage.csv
        one row per (pool, target): which method/model cells exist;
    results/experiments/<job>/agreement_<pool>.csv
        per-cell new-vs-legacy deltas for every pool that has a legacy table.

Nothing is written into ``results/`` itself unless ``--promote`` is given; the
legacy tables stay the reference until the new run is verified.

Usage
-----
    python scripts/collect_results.py --job main
    python scripts/collect_results.py --job main --promote
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
POOLS = ["aeeem", "tera", "nasa", "relink"]
LEGACY_METHODS = ["AEFS", "Ablation_no2D", "AllFeatures", "FilterMI", "WrapperRFE"]
LEGACY_MODELS = ["lightgbm", "random_forest"]
KEYS = ["pool", "target", "method", "model"]
SCORES = ["n_features", "mean_jaccard", "test_mcc", "test_f1", "test_auc_roc"]


def load(job: str) -> pd.DataFrame:
    path = ROOT / "results" / "experiments" / job / "all_rows.csv"
    if not path.exists():
        raise SystemExit(f"missing {path} - run 'python scripts/fanout.py merge "
                         f"--job {job}' first")
    df = pd.read_csv(path)
    missing = [c for c in KEYS + SCORES if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns {missing}")
    return df


def split_pools(df: pd.DataFrame, job: str) -> Path:
    out = ROOT / "results" / "experiments" / job
    for pool in POOLS:
        sub = df[df.pool == pool]
        if len(sub):
            sub.to_csv(out / f"{pool}_results.csv", index=False,
                       lineterminator="\n")
    return out


def write_legacy_mirrors(df: pd.DataFrame, job: str, promote: bool) -> list:
    out = ROOT / "results" / "experiments" / job / "legacy"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    sub = df[df.method.isin(LEGACY_METHODS) & df.model.isin(LEGACY_MODELS)]
    for pool in POOLS:
        legacy = ROOT / "results" / f"{pool}_stage4_full_results.csv"
        if not legacy.exists():
            continue
        have = sub[sub.pool == pool]
        if not len(have):
            continue
        path = out / legacy.name
        have.to_csv(path, index=False, lineterminator="\n")
        if promote:
            have.to_csv(legacy, index=False, lineterminator="\n")
        written.append(path)
    return written


def coverage(df: pd.DataFrame, job: str) -> pd.DataFrame:
    rows = []
    for (pool, target), g in df.groupby(["pool", "target"]):
        rows.append(dict(pool=pool, target=target, rows=len(g),
                         methods="|".join(sorted(g.method.unique())),
                         models="|".join(sorted(g.model.unique()))))
    cov = pd.DataFrame(rows).sort_values(["pool", "target"])
    cov.to_csv(ROOT / "results" / "experiments" / job / "coverage.csv",
               index=False, lineterminator="\n")
    return cov


def agreement(df: pd.DataFrame, job: str) -> pd.DataFrame:
    frames = []
    for pool in POOLS:
        legacy_path = ROOT / "results" / f"{pool}_stage4_full_results.csv"
        if not legacy_path.exists():
            continue
        old = pd.read_csv(legacy_path)
        new = df[(df.pool == pool) & df.method.isin(LEGACY_METHODS)
                 & df.model.isin(LEGACY_MODELS)]
        if "seed" in new.columns and new.seed.nunique() > 1:
            # multi-seed runs: average over seeds before comparing with the
            # single-seed legacy tables
            new = (new.groupby(KEYS, as_index=False)
                   .agg({"n_features": "mean", "mean_jaccard": "mean",
                         "test_mcc": "mean", "test_f1": "mean",
                         "test_auc_roc": "mean"}))
        m = old.merge(new, on=KEYS, suffixes=("_legacy", "_new"))
        if not len(m):
            continue
        m["d_mcc"] = m.test_mcc_new - m.test_mcc_legacy
        m["d_auc"] = m.test_auc_roc_new - m.test_auc_roc_legacy
        m["d_f1"] = m.test_f1_new - m.test_f1_legacy
        m["d_nfeat"] = m.n_features_new - m.n_features_legacy
        m["d_jaccard"] = m.mean_jaccard_new - m.mean_jaccard_legacy
        m.to_csv(ROOT / "results" / "experiments" / job / f"agreement_{pool}.csv",
                 index=False, lineterminator="\n")
        frames.append(m.assign(pool=pool))
    if not frames:
        return pd.DataFrame()
    allm = pd.concat(frames, ignore_index=True)
    print(f"\n{'pool':<10} {'cells':>5} {'|dMCC|':>8} {'|dAUC|':>8} "
          f"{'|dF1|':>8} {'|dNfeat|':>9} {'|dJacc|':>8} {'same nfeat':>10}")
    for pool, g in allm.groupby("pool"):
        print(f"{pool:<10} {len(g):>5} {g.d_mcc.abs().mean():>8.4f} "
              f"{g.d_auc.abs().mean():>8.4f} {g.d_f1.abs().mean():>8.4f} "
              f"{g.d_nfeat.abs().mean():>9.3f} "
              f"{g.d_jaccard.abs().mean():>8.4f} "
              f"{(g.d_nfeat == 0).mean():>9.0%}")
    print(f"{'ALL':<10} {len(allm):>5} {allm.d_mcc.abs().mean():>8.4f} "
          f"{allm.d_auc.abs().mean():>8.4f} {allm.d_f1.abs().mean():>8.4f} "
          f"{allm.d_nfeat.abs().mean():>9.3f} "
          f"{allm.d_jaccard.abs().mean():>8.4f} "
          f"{(allm.d_nfeat == 0).mean():>9.0%}")
    return allm


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", required=True,
                    help="experiment id, e.g. main / ablation / cpdp / multiseed")
    ap.add_argument("--promote", action="store_true",
                    help="also overwrite results/<pool>_stage4_full_results.csv")
    args = ap.parse_args(argv)

    df = load(args.job)
    print(f"[collect] {args.job}: {len(df)} rows, "
          f"{df.pool.nunique()} pools, {df.target.nunique()} targets, "
          f"{df.method.nunique()} methods, {df.model.nunique()} models")
    out = split_pools(df, args.job)
    cov = coverage(df, args.job)
    dup_keys = KEYS + (["seed"] if "seed" in df.columns else [])
    dup = df.duplicated(dup_keys).sum()
    print(f"[collect] coverage: {len(cov)} targets, missing-method targets: "
          f"{int((cov.methods.str.split('|').map(len) < df.method.nunique()).sum())}"
          f", duplicate key rows: {dup}")
    mirrors = write_legacy_mirrors(df, args.job, args.promote)
    for p in mirrors:
        print(f"[collect] legacy mirror {'promoted to ' if args.promote else ''}"
              f"{p.relative_to(ROOT)}")
    agreement(df, args.job)
    print(f"[collect] outputs in {out.relative_to(ROOT)}")
    if not args.promote:
        print("[collect] legacy tables untouched (use --promote to overwrite)")


if __name__ == "__main__":
    main()
