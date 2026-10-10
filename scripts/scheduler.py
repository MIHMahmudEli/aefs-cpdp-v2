"""Experiment registry and dependency-aware watchdog.

Keeps ``results/registry.json`` up to date so every experiment has exactly one
machine-readable state:

    PENDING -> RUNNING -> COMPLETED -> FAILED -> RETRY -> VERIFIED
                                                   -> USED_IN_MANUSCRIPT

Usage
-----
    python scripts/scheduler.py status
    python scripts/scheduler.py next [--workers 16]
    python scripts/scheduler.py mark <id> RUNNING --worker w1 --runtime 120
    python scripts/scheduler.py mark <id> COMPLETED --result path/to.csv
    python scripts/scheduler.py mark <id> FAILED --reason "empty shard"
    python scripts/scheduler.py verify <id>
    python scripts/scheduler.py retry <id>

Rules (mirrors docs/EXPERIMENTS.md section 2):

1. an experiment is only scheduled if every dependency is COMPLETED or VERIFIED;
2. an experiment never runs twice while a RUNNING record exists;
3. a shard that produced no output rows is FAILED, never COMPLETED;
4. COMPLETED requires the expected output file to exist and be non-empty;
5. VERIFIED requires the verification test to pass;
6. retries are capped (3) with the reason recorded.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "results" / "registry.json"
MAX_ATTEMPTS = 3

# id -> (depends_on, result file produced when COMPLETED)
EXPERIMENTS = {
    "EXP-001-V": ([], ROOT / "results" / "experiments" / "validate" / "all_rows.csv"),
    "EXP-001": (["EXP-001-V"], ROOT / "results" / "experiments" / "main" / "all_rows.csv"),
    "EXP-002": (["EXP-001-V"], ROOT / "results" / "experiments" / "ablation" / "all_rows.csv"),
    "EXP-003": (["EXP-001-V"], ROOT / "results" / "experiments" / "multiseed" / "all_rows.csv"),
    "EXP-004": (["EXP-001-V"], ROOT / "results" / "experiments" / "cpdp" / "all_rows.csv"),
    "EXP-005": (["EXP-001"], ROOT / "results" / "experiments" / "sensitivity_nasa_relink" / "all_rows.csv"),
    "ANL-001": (["EXP-001"], ROOT / "results" / "analysis" / "descriptive.csv"),
    "ANL-002": (["EXP-001"], ROOT / "results" / "analysis" / "rq1_rq2.csv"),
    "ANL-003": (["EXP-001", "EXP-002"], ROOT / "results" / "analysis" / "rq3.csv"),
    "ANL-004": (["EXP-003"], ROOT / "results" / "analysis" / "rq4.csv"),
    "ANL-005": (["EXP-001", "EXP-004"], ROOT / "results" / "analysis" / "rq5.csv"),
    "ANL-006": (["EXP-001"], ROOT / "results" / "analysis" / "rq6.csv"),
    "MS-006": (["ANL-001", "ANL-002"], ROOT / "results" / "analysis" / "tables_regenerated.flag"),
    "MS-007": (["MS-006"], ROOT / "paper-ist" / "main.pdf"),
}


def load() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {eid: {"id": eid, "status": "PENDING", "attempts": 0,
                  "depends_on": list(deps), "result": str(path)}
            for eid, (deps, path) in EXPERIMENTS.items()}


def save(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    reg["_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def _public(reg: dict) -> dict:
    return {k: v for k, v in reg.items() if not k.startswith("_")}


def cmd_status(_) -> None:
    reg = load()
    print(f"{'ID':<12} {'status':<18} {'try':>3} {'deps':<24} result")
    for eid in EXPERIMENTS:
        e = reg.get(eid, {"status": "PENDING", "attempts": 0})
        deps = ",".join(EXPERIMENTS[eid][0]) or "-"
        res = e.get("result", "")
        res = Path(res).name if res else ""
        print(f"{eid:<12} {e.get('status','PENDING'):<18} "
              f"{e.get('attempts',0):>3} {deps:<24} {res}")


def _ready(reg: dict, eid: str) -> bool:
    for dep in EXPERIMENTS[eid][0]:
        if reg.get(dep, {}).get("status") not in ("COMPLETED", "VERIFIED"):
            return False
    return True


def cmd_next(args) -> None:
    reg = load()
    out = []
    for eid, (deps, _) in EXPERIMENTS.items():
        e = reg.get(eid, {"status": "PENDING", "attempts": 0})
        if e["status"] in ("PENDING", "FAILED", "RETRY") and _ready(reg, eid):
            if e.get("attempts", 0) >= MAX_ATTEMPTS and e["status"] == "FAILED":
                continue
            out.append(eid)
    print("\n".join(out[: args.workers]) or "(nothing ready)")
    if out:
        print(f"# {len(out)} ready, running up to {args.workers}", file=sys.stderr)


def cmd_mark(args) -> None:
    reg = load()
    eid = args.id
    if eid not in EXPERIMENTS:
        raise SystemExit(f"unknown experiment {eid}")
    e = reg.setdefault(eid, {"id": eid, "status": "PENDING", "attempts": 0,
                             "depends_on": list(EXPERIMENTS[eid][0])})
    status = args.status.upper()
    if status not in ("PENDING", "RUNNING", "COMPLETED", "FAILED",
                      "RETRY", "VERIFIED", "USED_IN_MANUSCRIPT"):
        raise SystemExit(f"bad status {status}")
    if status == "RUNNING":
        if any(k == eid and v.get("status") == "RUNNING"
               for k, v in _public(reg).items()):
            raise SystemExit(f"{eid} is already RUNNING (duplicate-run guard)")
        if not _ready(reg, eid):
            raise SystemExit(f"{eid} dependencies not satisfied")
        e["attempts"] = e.get("attempts", 0) + 1
        e["started"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    e["status"] = status
    if args.worker:
        e["workers"] = args.worker
    if args.runtime:
        e["runtime_s"] = args.runtime
    if args.result:
        e["result"] = args.result
    if args.reason:
        e["reason"] = args.reason
        if status == "FAILED" and e["attempts"] >= MAX_ATTEMPTS:
            e["status"] = "FAILED"
    e["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    if status == "COMPLETED":
        path = Path(args.result) if args.result else Path(EXPERIMENTS[eid][1])
        if not path.exists() or path.stat().st_size == 0:
            e["status"] = "FAILED"
            e["reason"] = f"output missing or empty: {path}"
            print(f"{eid} -> FAILED ({e['reason']})", file=sys.stderr)
        if "empty" in (args.reason or ""):
            e["status"] = "FAILED"
    save(reg)
    print(f"{eid} -> {e['status']} (attempt {e.get('attempts',0)})")


def verify_exp001v() -> tuple[bool, str]:
    """Compare the canonical runner's aeeem output with the legacy table."""
    import pandas as pd
    new = ROOT / "results" / "experiments" / "validate" / "all_rows.csv"
    old = ROOT / "results" / "aeeem_stage4_full_results.csv"
    if not new.exists() or not old.exists():
        return False, "input missing"
    a, b = pd.read_csv(old), pd.read_csv(new)
    m = a.merge(b, on=["pool", "target", "method", "model"], suffixes=("_l", "_n"))
    if len(m) != len(a):
        return False, f"only {len(m)}/{len(a)} cells overlap"
    feats = (m.n_features_l - m.n_features_n).abs()
    mcc = (m.test_mcc_l - m.test_mcc_n).abs()
    ok_feat = (feats <= 1e-9).mean()
    ok_mcc = (mcc <= 0.05).mean()
    checks = [
        (ok_feat >= 0.90, f"n_features identical on {ok_feat:.0%} (need >=90%)"),
        (ok_mcc >= 0.80, f"MCC within 0.05 on {ok_mcc:.0%} (need >=80%)"),
        (mcc.mean() < 0.03, f"mean abs dMCC {mcc.mean():.4f} (need <0.03)"),
    ]
    report = "; ".join(msg for _, msg in checks)
    return all(ok for ok, _ in checks), report


def verify_exp001() -> tuple[bool, str]:
    """Coverage check for the main experiment (all four pools)."""
    import pandas as pd
    path = ROOT / "results" / "experiments" / "main" / "all_rows.csv"
    if not path.exists():
        return False, f"missing {path}"
    df = pd.read_csv(path)
    problems = []
    pools = set(df.pool)
    if pools != {"aeeem", "tera", "nasa", "relink"}:
        problems.append(f"pools={sorted(pools)}")
    keys = ["pool", "target", "method", "model"]
    dup = int(df.duplicated(keys).sum())
    if dup:
        problems.append(f"{dup} duplicate (pool,target,method,model) rows")
    bad = df[~df.test_mcc.between(-1, 1) | ~df.test_auc_roc.between(0, 1)]
    if len(bad):
        problems.append(f"{len(bad)} rows with out-of-range scores")
    expected = {"tera": {"lightgbm", "random_forest"},
                "aeeem": {"lightgbm", "random_forest", "svm"},
                "nasa": {"lightgbm", "random_forest", "svm"},
                "relink": {"lightgbm", "random_forest", "svm"}}
    counts = df.groupby("pool").target.nunique().to_dict()
    methods = set(df.method.unique())
    for pool, g in df.groupby("pool"):
        got = set(g.model.unique())
        if got != expected[pool]:
            problems.append(f"{pool} models={sorted(got)}")
        for target, t in g.groupby("target"):
            if set(t.method) != methods:
                problems.append(f"{pool}/{target} methods={sorted(set(t.method))}")
                break
    report = (f"{len(df)} rows, {counts} targets/pool, methods={sorted(methods)}")
    return (not problems), (report if not problems else
                            report + "; " + "; ".join(problems))


def verify_exp005() -> tuple[bool, str]:
    """EXP-005: SHAP / stability sweeps, convergence, runtime for nasa+relink."""
    import pandas as pd
    path = ROOT / "results" / "experiments" / "sensitivity_nasa_relink" / "all_rows.csv"
    if not path.exists():
        return False, f"missing {path}"
    df = pd.read_csv(path, dtype={"setting": str})
    problems = []
    if set(df.pool) != {"nasa", "relink"}:
        problems.append(f"pools={sorted(set(df.pool))}")
    counts = df.groupby("pool").target.nunique().to_dict()
    if counts != {"nasa": 4, "relink": 3}:
        problems.append(f"targets={counts}")
    got = set(df["experiment"])
    want = {"shap", "stability", "convergence", "runtime"}
    if got != want:
        problems.append(f"experiments={sorted(got)}")
    shap = set(df[df["experiment"] == "shap"].setting)
    if shap != {"0.01", "0.02", "0.03", "0.05"}:
        problems.append(f"shap settings={sorted(shap)}")
    stab = set(df[df["experiment"] == "stability"].setting)
    if stab != {"0.2", "0.3", "0.4", "0.5"}:
        problems.append(f"stability settings={sorted(stab)}")
    keys = ["pool", "target", "experiment", "setting", "method"]
    if df.duplicated(keys).sum():
        problems.append(f"{int(df.duplicated(keys).sum())} duplicate rows")
    report = (f"{len(df)} rows, {counts} targets/pool, "
              f"experiments={sorted(got)}")
    return (not problems), (report if not problems else
                            report + "; " + "; ".join(problems))


GRID_EXPECTED = {
    "EXP-002": ("ablation",
                ["Ablation_noRUP", "Ablation_noSFS", "Ablation_noSHAP"]),
    "EXP-003": ("multiseed", ["AEFS"]),
    "EXP-004": ("cpdp", ["MI_DS", "MI_TW"]),
}


def verify_grid(eid: str) -> tuple[bool, str]:
    """Coverage check for the sharded follow-up experiments."""
    import pandas as pd
    job, methods = GRID_EXPECTED[eid]
    path = ROOT / "results" / "experiments" / job / "all_rows.csv"
    if not path.exists():
        return False, f"missing {path}"
    df = pd.read_csv(path)
    problems = []
    keys = ["pool", "target", "method", "model"] + (
        ["seed"] if "seed" in df.columns else [])
    dup = int(df.duplicated(keys).sum())
    if dup:
        problems.append(f"{dup} duplicate rows")
    counts = df.groupby("pool").target.nunique().to_dict()
    if counts.get("tera") != 68:
        problems.append(f"tera targets={counts.get('tera')} (expected 68)")
    if set(df.method.unique()) != set(methods):
        problems.append(f"methods={sorted(set(df.method.unique()))}")
    if not df.test_mcc.between(-1, 1).all():
        problems.append("out-of-range MCC")
    if eid == "EXP-003":
        got = set(int(s) for s in df.get("seed", pd.Series([], dtype=int)).unique())
        want = {42, 7, 123, 2024, 555}
        if not got.issubset(want):
            problems.append(f"unexpected seeds {sorted(got)}")
        missing = want - got
        if missing:
            problems.append(f"missing seeds {sorted(missing)}")
    report = f"{len(df)} rows, {counts} targets/pool, methods={sorted(set(df.method))}"
    return (not problems), (report if not problems else report + "; " + "; ".join(problems))


def cmd_verify(args) -> None:
    reg = load()
    eid = args.id
    if eid == "EXP-001-V":
        ok, report = verify_exp001v()
    elif eid == "EXP-001":
        ok, report = verify_exp001()
    elif eid == "EXP-005":
        ok, report = verify_exp005()
    elif eid in GRID_EXPECTED:
        ok, report = verify_grid(eid)
    else:
        path = Path(EXPERIMENTS[eid][1])
        ok, report = path.exists() and path.stat().st_size > 0, f"{path.name} exists"
    if not ok:
        raise SystemExit(f"{eid} VERIFY FAILED: {report}")
    reg.setdefault(eid, {"id": eid, "attempts": 0})["status"] = "VERIFIED"
    reg[eid]["verified"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    reg[eid]["verification"] = report
    save(reg)
    print(f"{eid} VERIFIED: {report}")


def cmd_retry(args) -> None:
    reg = load()
    e = reg.setdefault(args.id, {"id": args.id, "attempts": 0,
                                 "status": "PENDING"})
    e["status"] = "RETRY"
    save(reg)
    print(f"{args.id} -> RETRY")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status").set_defaults(func=cmd_status)
    p = sub.add_parser("next")
    p.add_argument("--workers", type=int, default=16)
    p.set_defaults(func=cmd_next)
    p = sub.add_parser("mark")
    p.add_argument("id")
    p.add_argument("status")
    p.add_argument("--worker")
    p.add_argument("--runtime", type=int)
    p.add_argument("--result")
    p.add_argument("--reason")
    p.set_defaults(func=cmd_mark)
    p = sub.add_parser("verify")
    p.add_argument("id")
    p.set_defaults(func=cmd_verify)
    p = sub.add_parser("retry")
    p.add_argument("id")
    p.set_defaults(func=cmd_retry)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
