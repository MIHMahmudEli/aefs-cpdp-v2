"""Sharded experiment runner with full provenance.

Every invocation writes

    <out>/rows_s<NN>.csv       this shard's rows (never overwrites another shard)
    <out>/config.json          the exact configuration that produced them
    <out>/metadata.json        environment, dataset checksums, code version,
                               timing, row count, completion flag

so a result can always be traced back to
configuration -> code -> dataset -> worker -> output.

Usage:
    python scripts/run_experiment.py --experiment main --pools aeeem nasa relink
    python scripts/run_experiment.py --experiment main --pools tera --shard 3 --nshards 16
    python scripts/run_experiment.py --experiment multiseed --pools aeeem tera --seeds 42 7

On Kaggle the output directory is /kaggle/working/out and, because script
kernels expose no files through the API, every CSV is also printed between
``===== name.csv (n bytes) =====`` markers so it can be read back from the
kernel log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import aefs_core as core  # noqa: E402
import regen_sensitivity as rs  # noqa: E402

ALL_POOLS = ["aeeem", "tera", "nasa", "relink"]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or None
    except Exception:
        return None


def dataset_fingerprint(pools: list) -> dict:
    """md5 over each pool's CSVs - changes iff the input data change."""
    out = {}
    for pool in pools:
        d = core.RAW_DIRS.get(pool)
        if d is None or not d.is_dir():
            continue
        h = hashlib.md5()
        files = sorted(d.glob("*.csv"))
        for f in files:
            h.update(f.name.encode())
            h.update(f.read_bytes())
        out[pool] = dict(files=len(files), md5=h.hexdigest())
    return out


def environment_info() -> dict:
    info = dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        machine=platform.machine(),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        hostname=platform.node(),
        git_commit=git_commit(),
    )
    for mod in ("numpy", "pandas", "scipy", "sklearn", "lightgbm",
                "imblearn", "shap"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception as exc:  # pragma: no cover
            info[mod] = f"unavailable ({type(exc).__name__})"
    return info


def dump_block(path: Path) -> None:
    """Print a file between ===== markers (Kaggle has no file API)."""
    data = path.read_bytes()
    print(f"===== {path.name} ({len(data)} bytes) =====", flush=True)
    sys.stdout.write(data.decode("utf-8", "replace"))
    if not data.endswith(b"\n"):
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True, choices=sorted(core.EXPERIMENTS))
    ap.add_argument("--pools", nargs="+", default=ALL_POOLS, choices=ALL_POOLS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="seeds for --experiment multiseed")
    ap.add_argument("--out", default=None,
                    help="output directory (default results/experiments/<id>)")
    ap.add_argument("--experiment-id", default=None,
                    help="registry id, default <experiment>")
    ap.add_argument("--stdout", action="store_true",
                    help="also print every output CSV (for Kaggle logs)")
    args = ap.parse_args()

    exp_id = args.experiment_id or args.experiment
    out = Path(args.out) if args.out else ROOT / "results" / "experiments" / exp_id
    out.mkdir(parents=True, exist_ok=True)

    cfg = dict(
        experiment=args.experiment,
        experiment_id=exp_id,
        pools=args.pools,
        shard=args.shard,
        nshards=args.nshards,
        seeds=args.seeds or (core.DEFAULT_SEEDS if args.experiment == "multiseed" else None),
        seed=rs.RANDOM_STATE,
        methods=(core.MAIN_METHODS if args.experiment == "main" else
                 core.ABLATION_METHODS if args.experiment == "ablation" else
                 core.CPDP_METHODS if args.experiment == "cpdp" else
                 core.SENS_METHODS if args.experiment == "sensitivity"
                 else ["AEFS"]),
        models={p: core.models_for(p) for p in args.pools},
        thresholds=dict(variance=1e-4, relevance=0.02, redundancy=0.90,
                        sfs_min_improvement=0.001, shap_mass=0.02,
                        stability=0.30, min_features=5),
        data_fingerprint=dataset_fingerprint(args.pools),
    )
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(f"[run] experiment={args.experiment} id={exp_id} pools={args.pools} "
          f"shard={args.shard}/{args.nshards} out={out}", flush=True)
    print(f"[run] data root: {core._DATA_ROOT}", flush=True)
    for pool in args.pools:
        d = core.RAW_DIRS.get(pool)
        n = len(list(d.glob("*.csv"))) if d is not None and d.is_dir() else -1
        print(f"[run]   {pool}: {d} -> {n} csv", flush=True)
    print(f"[run] SHAP backend: "
          f"{'shap.TreeExplainer' if rs._HAVE_SHAP else 'lightgbm pred_contrib'}",
          flush=True)

    t0 = time.time()
    runner = core.EXPERIMENTS[args.experiment]
    if args.experiment == "multiseed":
        df = runner(args.pools, args.shard, args.nshards, seeds=cfg["seeds"])
    else:
        df = runner(args.pools, args.shard, args.nshards)
    elapsed = time.time() - t0

    rows_path = out / f"rows_s{args.shard:02d}.csv"
    df.to_csv(rows_path, index=False, lineterminator="\n")

    meta = dict(
        experiment_id=exp_id,
        shard=args.shard,
        nshards=args.nshards,
        rows=len(df),
        elapsed_s=round(elapsed, 1),
        complete=bool(len(df)),
        output=rows_path.name,
        environment=environment_info(),
        targets=(sorted(df["target"].unique()) if len(df) else []),
        pools=sorted(df["pool"].unique()) if len(df) else [],
    )
    (out / f"metadata_s{args.shard:02d}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[run] {len(df)} rows in {elapsed:.1f}s -> {rows_path.name}", flush=True)
    if args.stdout:
        dump_block(rows_path)
    print("EXPERIMENT_COMPLETE" if len(df) else "EXPERIMENT_EMPTY", flush=True)


if __name__ == "__main__":
    main()
