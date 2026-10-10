"""Fan experiments out to the 16 Kaggle workers recorded in ``.env``.

Kaggle script kernels accept no CLI arguments and ``kaggle kernels push``
uploads only the code file, so every worker receives:

  * its own private assets dataset  <worker>/aefs-shard-assets
      aefs_core.py, regen_sensitivity.py, run_experiment.py, data/ (84 CSVs)
  * a private kernel                 <worker>/<kernel slug>
      run.py, which hard-codes the experiment id, pool list, shard index and
      shard count, then prints every produced CSV between
      ``===== name.csv (n bytes) =====`` markers (Kaggle's file API reports
      ``files: []`` for script kernels, so results come back through the log)

Authentication: ``kagglesdk`` reads ``KAGGLE_API_TOKEN`` and ignores
``KAGGLE_CONFIG_DIR``, so every command sets the token per worker.  A worker
whose token is not set would silently act as the default identity - hence the
``tokens`` subcommand and the username check.

Subcommands
-----------
tokens   validate every credential in .env
setup    (re)build the per-worker assets dataset
push     push the per-worker kernel (skips workers already complete)
poll     bounded wait for every kernel
fetch    save each kernel log to logs_<job>/<worker>.log
merge    parse the logs -> results/experiments/<job>/rows_sNN.csv + all_rows.csv
run      setup + push + poll + fetch + merge in one go
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOG_ROOT = ROOT / "logs"
EXP_ROOT = ROOT / "results" / "experiments"
DATASET_SLUG = "aefs-shard-assets"
ALL_POOLS = ["aeeem", "tera", "nasa", "relink"]

# authoring environment (docs/ENVIRONMENT.md); installed before any scientific
# import so a Kaggle run is directly comparable with the local numbers.
PINNED = ["numpy==2.5.1", "pandas==3.0.5", "scipy==1.18.1",
          "scikit-learn==1.9.0", "lightgbm==4.7.0", "imbalanced-learn==0.14.2"]

JOBS: dict = {
    # EXP-001-V: canonical runner vs the legacy AEEEM table (1 worker, no sharding)
    "validate": dict(experiment="main", pools=["aeeem"], nshards=1, nworkers=1,
                     seeds=None, pin=True, timeout=60),
    # EXP-001: 6 methods x 3 classifiers x 4 pools (80 LOPO targets)
    "main": dict(experiment="main", pools=ALL_POOLS, nshards=16, nworkers=16,
                 seeds=None, pin=True, timeout=240),
    # EXP-002: component ablation
    "ablation": dict(experiment="ablation", pools=ALL_POOLS, nshards=16,
                     nworkers=16, seeds=None, pin=True, timeout=240),
    # EXP-004: CPDP-aware filter baselines
    "cpdp": dict(experiment="cpdp", pools=ALL_POOLS, nshards=16, nworkers=16,
                 seeds=None, pin=True, timeout=240),
    # EXP-003: 5 seeds x 4 pools x 3 classifiers
    "multiseed": dict(experiment="multiseed", pools=ALL_POOLS, nshards=16,
                      nworkers=16, seeds=[42, 7, 123, 2024, 555], pin=True,
                      timeout=300),
    # EXP-005: SHAP / stability sweeps, convergence and runtime for the two
    # pools that never had them (audit A10)
    "sensitivity": dict(experiment="sensitivity", pools=["nasa", "relink"],
                        nshards=16, nworkers=16, seeds=None, pin=True,
                        timeout=180, out="sensitivity_nasa_relink"),
}

CURRENT_JOB = "main"
_PROBE = ""
SKIP_USERS = {"leyamridha"}   # dataset versions return 403 (storage quota)
TOKENS: dict = {}
ENV_FILE = ROOT / ".env"


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------
def load_tokens() -> list:
    """Worker credentials from the repo's .env (never logged, never uploaded)."""
    if not ENV_FILE.exists():
        raise SystemExit(f"missing {ENV_FILE}")
    out, user = [], None
    TOKENS.clear()
    for line in ENV_FILE.read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, val = (part.strip() for part in s.split("=", 1))
        if not val:
            continue
        if key == "HF_TOKEN" or key.lower() == "username":
            if key.lower() == "username":
                user = val
            continue
        if key.lower().startswith("kaggel_tok") or key.lower().startswith("kaggle_key"):
            if user and user not in SKIP_USERS:
                TOKENS[user] = val
                out.append((user, val))
            user = None
    if not out:
        raise SystemExit("no worker credentials parsed from .env")
    print(f"[creds] {len(out)} workers from .env")
    return out


def _env(user: str) -> dict:
    env = dict(os.environ)
    env["KAGGLE_API_TOKEN"] = TOKENS[user]
    env["KAGGLE_CONFIG_DIR"] = str(ROOT / "logs" / "kcfg" / user)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def kaggle(user: str, *args: str, timeout: int = 900) -> tuple:
    p = subprocess.run(["kaggle", *args], env=_env(user), capture_output=True,
                       timeout=timeout, cwd=str(ROOT))
    return (p.returncode,
            (p.stdout or b"").decode("utf-8", "replace"),
            (p.stderr or b"").decode("utf-8", "replace"))


def job_cfg() -> dict:
    return JOBS[CURRENT_JOB]


def active() -> list:
    """Workers that will actually run this job.

    The shard count is derived from this list: a shard index that no
    worker owns would silently never execute.
    """
    load_tokens()
    return list(TOKENS)[: job_cfg()["nworkers"]]


def nshards() -> int:
    return len(active())


def slug() -> str:
    return f"aefs-exp-{CURRENT_JOB}"


def title() -> str:
    # Kaggle derives the kernel slug from the title, so keep them aligned:
    # "AEFS exp main" -> aefs-exp-main
    return f"AEFS exp {CURRENT_JOB}"


def logs_dir() -> Path:
    d = LOG_ROOT / CURRENT_JOB
    d.mkdir(parents=True, exist_ok=True)
    return d


def exp_id() -> str:
    """Directory name of the merged output for the current job."""
    return job_cfg().get("out", CURRENT_JOB)


def exp_dir() -> Path:
    d = EXP_ROOT / exp_id()
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------
def cmd_tokens(_) -> None:
    ok = 0
    for user, _tok in load_tokens():
        rc, out, err = kaggle(user, "config", "view")
        m = re.search(r"username:\s*(\S+)", out)
        got = m.group(1) if m else "?"
        good = rc == 0 and got == user
        ok += good
        print(f"{user:<24} {'OK' if good else 'FAIL'} -> {got}"
              f"{'' if good else ' :: ' + (err.strip() or out.strip())[:100]}")
    print(f"{ok}/{len(TOKENS)} tokens verified")


def _build_assets(user: str, folder: Path) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for name in ("aefs_core.py", "regen_sensitivity.py", "run_experiment.py"):
        shutil.copy2(HERE / name, folder / name)
    with __import__("zipfile").ZipFile(ROOT / "AEFS_data.zip") as z:
        z.extractall(folder / "data")
    (folder / "dataset-metadata.json").write_text(
        '{\n  "title": "AEFS shard assets",\n'
        f'  "id": "{user}/{DATASET_SLUG}",\n'
        '  "licenses": [{"name": "CC0-1.0"}]\n}\n', encoding="ascii")


def _dataset_exists(user: str) -> bool:
    rc, out, _ = kaggle(user, "datasets", "files", f"{user}/{DATASET_SLUG}",
                        "--page-size", "5")
    return rc == 0 and "creationDate" in out


_PROBE = ""
_MARKERS: dict = {}


def _wait_dataset(user: str, probe: str, timeout: int = 1800) -> bool:
    """Kaggle creates dataset versions asynchronously; a kernel pushed too
    soon mounts the *previous* version and silently loads the wrong data.

    Readiness is probed with a marker file whose name is unique to this
    upload, so an older version can never satisfy the check.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        rc, out, _ = kaggle(user, "datasets", "files", f"{user}/{DATASET_SLUG}",
                            "--page-size", "300")
        if rc == 0 and probe in out and "data/aeeem/eq.csv" in out:
            return True
        time.sleep(20)
    return False


def _upload_assets(user: str) -> int:
    folder = ROOT / "logs" / "kds" / user
    _build_assets(user, folder)
    probe = f"marker_{CURRENT_JOB}_{int(time.time())}.txt"
    (folder / probe).write_text(
        f"job={CURRENT_JOB}\nuser={user}\ntime={time.time()}\n",
        encoding="utf-8")
    _MARKERS[user] = probe
    # --dir-mode zip is required: the CLI *skips* sub-folders by default
    # ("Skipping folder: data; use '--dir-mode' to upload folders"), which
    # silently publishes a dataset with no CSV files at all.
    if _dataset_exists(user):
        rc, out, err = kaggle(user, "datasets", "version", "-p", str(folder),
                              "-m", f"refresh {CURRENT_JOB}", "-q",
                              "--dir-mode", "zip")
    else:
        rc, out, err = kaggle(user, "datasets", "create", "-p", str(folder),
                              "-q", "--dir-mode", "zip")
    print(f"{user:<24} upload rc={rc} {(out + err).strip()[-150:]}", flush=True)
    return rc


def cmd_setup(_) -> None:
    users = active()
    failed = []
    for user in users:
        if _upload_assets(user) != 0:
            failed.append(user)
    for user in failed:                       # one retry for transient errors
        print(f"{user:<24} retrying upload", flush=True)
        if _upload_assets(user) != 0:
            raise SystemExit(f"{user}: dataset upload failed twice")
    for user in users:
        ready = _wait_dataset(user, _MARKERS[user])
        print(f"{user:<24} ready={ready} probe={_MARKERS[user]}", flush=True)
        if not ready:
            raise SystemExit(f"{user}: dataset version did not become ready")


def _write_kernel(user: str, shard: int) -> Path:
    job = job_cfg()
    folder = ROOT / "logs" / "kern" / user
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    pools = job["pools"]
    seeds = job["seeds"]
    if job.get("pin"):
        pin_block = f'''
# Install the authoring library versions BEFORE any scientific import:
# pip replaces files on disk, but sys.modules would keep serving the old
# numpy for the rest of the process, so the versions are verified in a
# fresh interpreter (same procedure as the documented "pinned" job).
import subprocess as _sp, sys as _sys
_PIN = {PINNED!r}
print("[pin] installing:", " ".join(_PIN), flush=True)
_rc = _sp.run([_sys.executable, "-m", "pip", "install", "-q", *_PIN],
              capture_output=True, text=True)
print("[pin] pip rc=%s %s" % (_rc.returncode, ((_rc.stdout or "") + (_rc.stderr or ""))[-500:]), flush=True)
if _rc.returncode != 0:
    raise SystemExit("[pin] pip failed")
_chk = _sp.run([_sys.executable, "-c",
                "import numpy,pandas,sklearn,lightgbm,scipy,imblearn;"
                "print(numpy.__version__, pandas.__version__, sklearn.__version__,"
                " lightgbm.__version__, scipy.__version__, imblearn.__version__)"],
               capture_output=True, text=True)
_got = (_chk.stdout or "").strip().split()
_want = ["2.5.1", "3.0.5", "1.9.0", "4.7.0", "1.18.1", "0.14.2"]
if _chk.returncode != 0 or _got != _want:
    raise SystemExit("[pin] version check failed got=%r want=%r" % (_got, _want))
print("[pin] versions OK: " + " ".join(_got), flush=True)
'''
    else:
        pin_block = ""
    run = f'''"""AEFS {CURRENT_JOB} shard {shard} (worker {user})."""
import glob
import os
import sys

candidates = sorted(glob.glob("/kaggle/input/**/run_experiment.py", recursive=True))
if not candidates:
    tree = {{}}
    for root, dirs, files in os.walk("/kaggle/input"):
        tree[root] = sorted(files)[:20]
    raise SystemExit(f"assets dataset not mounted; input tree={{tree}}")
assets = None
for cand in candidates:
    if os.path.basename(cand) == "run_experiment.py":
        assets = os.path.dirname(cand)
        break
if assets is None:
    raise SystemExit(f"assets dataset not mounted; found={{candidates}}")
sys.path.insert(0, assets)
{pin_block}
import run_experiment

sys.argv = ["run_experiment.py",
            "--experiment", {job["experiment"]!r},
            "--pools", *{pools!r},
            "--shard", str({shard}),
            "--nshards", str({nshards()!r}),
            "--out", "/kaggle/working/out",
            "--experiment-id", {exp_id()!r},
            "--stdout"]
if {seeds!r} is not None:
    sys.argv += ["--seeds"] + [str(s) for s in {seeds!r}]
run_experiment.main()
'''
    (folder / "run.py").write_text(run, encoding="utf-8")
    datasets = f'["{user}/{DATASET_SLUG}"]'
    (folder / "kernel-metadata.json").write_text(
        '{\n'
        f'  "id": "{user}/{slug()}",\n'
        f'  "title": "{title()}",\n'
        '  "code_file": "run.py",\n'
        '  "language": "python",\n'
        '  "kernel_type": "script",\n'
        '  "is_private": "true",\n'
        '  "enable_gpu": "false",\n'
        '  "enable_tpu": "false",\n'
        '  "enable_internet": "true",\n'
        '  "machine_shape": "",\n'
        f'  "dataset_sources": {datasets},\n'
        '  "competition_sources": [],\n'
        '  "kernel_sources": [],\n'
        '  "model_sources": []\n}\n', encoding="ascii")
    return folder


def _status(user: str) -> str:
    rc, out, err = kaggle(user, "kernels", "status", f"{user}/{slug()}")
    m = re.search(r'status "([^"]+)"', out)
    if m:
        return m.group(1).replace("KernelWorkerStatus.", "")
    return f"UNKNOWN rc={rc} {(err or out).strip()[:80]}"


def _log_text(user: str) -> str:
    rc, out, err = kaggle(user, "kernels", "logs", f"{user}/{slug()}", timeout=300)
    text = out if rc == 0 else err
    try:
        events = json.loads(text)
        text = "".join(e.get("data", "") for e in events)
    except Exception:
        pass
    return text


def _finished(user: str) -> bool:
    f = logs_dir() / f"{user}.log"
    if f.exists() and "EXPERIMENT_COMPLETE" in f.read_text(encoding="utf-8",
                                                           errors="replace"):
        return True
    return False


def cmd_push(args) -> None:
    load_tokens()
    users = active()
    for i, user in enumerate(users):
        if _finished(user) and not args.force:
            print(f"{i:<3} {user:<24} SKIP (already complete)", flush=True)
            continue
        folder = _write_kernel(user, i)
        rc, out, err = kaggle(user, "kernels", "push", "-p", str(folder))
        print(f"{i:<3} {user:<24} rc={rc} {(out + err).strip()[-170:]}", flush=True)


def cmd_poll(_) -> None:
    load_tokens()
    users = active()
    start = time.time()
    deadline = start + job_cfg()["timeout"] * 60
    done: dict = {}
    errs: dict = {}
    grace = 240
    while time.time() < deadline and len(done) < len(users):
        for user in users:
            if user in done:
                continue
            st = _status(user)
            if st in ("COMPLETE", "CANCELLED"):
                # right after a push the API still reports the previous run
                if time.time() - start < grace:
                    continue
                done[user] = st
                print(f"{user:<24} {st} ({len(done)}/{len(users)})", flush=True)
            elif st == "ERROR":
                # right after a push the API can still report the previous run
                if time.time() - start >= grace and errs.get(user):
                    done[user] = "ERROR"
                    print(f"{user:<24} ERROR ({len(done)}/{len(users)})", flush=True)
                errs[user] = True
        if len(done) < len(users):
            print(f"waiting on {len(users) - len(done)} ...", flush=True)
            time.sleep(45)
    for user in users:
        if user not in done:
            done[user] = "TIMEOUT"
            print(f"{user:<24} TIMEOUT")
    bad = [f"{u}:{s}" for u, s in done.items() if s != "COMPLETE"]
    print(f"complete={sum(1 for s in done.values() if 'COMPLETE' in s)}/"
          f"{len(users)} problems={bad}")


def cmd_fetch(_) -> None:
    load_tokens()
    users = active()
    for user in users:
        text = _log_text(user)
        dest = logs_dir() / f"{user}.log"
        dest.write_text(text, encoding="utf-8")
        complete = "EXPERIMENT_COMPLETE" in text
        empty = "EXPERIMENT_EMPTY" in text
        print(f"{user:<24} {len(text):>8} chars  complete={complete} empty={empty}")


def parse_blocks(text: str) -> dict:
    """Return {filename: body} from a kernel log's ===== markers."""
    header = re.compile(r"^===== (\S+?)(?: \((\d+) bytes\))? =====$")
    lines = text.splitlines(keepends=True)
    blocks, i = {}, 0
    while i < len(lines):
        m = header.match(lines[i].rstrip("\r\n"))
        if not m:
            i += 1
            continue
        name, size = m.group(1), m.group(2)
        i += 1
        buf, total = [], 0
        while i < len(lines):
            if header.match(lines[i].rstrip("\r\n")):
                break
            buf.append(lines[i])
            total += len(lines[i].encode("utf-8"))
            i += 1
            if size and total >= int(size):
                break
        blocks[name] = "".join(buf)
    return blocks


def cmd_merge(_) -> None:
    """Logs -> per-shard CSVs -> all_rows.csv (+ per-pool splits)."""
    import pandas as pd
    load_tokens()
    out = exp_dir()
    shards, problems = {}, []
    for user in active():
        f = logs_dir() / f"{user}.log"
        if not f.exists():
            problems.append(f"{user}: no log")
            continue
        blocks = parse_blocks(f.read_text(encoding="utf-8", errors="replace"))
        csvs = {k: v for k, v in blocks.items() if k.endswith(".csv")}
        if not csvs:
            problems.append(f"{user}: no CSV block")
            continue
        for name, body in csvs.items():
            shards[name] = body
            (out / name).write_text(body, encoding="utf-8", newline="")
        print(f"{user:<24} {sorted(csvs)}")

    if not shards:
        raise SystemExit("merge: no CSV blocks found")
    frames = []
    for name, body in sorted(shards.items()):
        from io import StringIO
        frames.append(pd.read_csv(StringIO(body)))
    df = pd.concat(frames, ignore_index=True)
    key = [c for c in ["pool", "target", "experiment", "setting", "method",
                       "model", "seed"] if c in df.columns]
    dupes = df.duplicated(subset=key).sum()
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(out / "all_rows.csv", index=False, lineterminator="\n")
    for pool in sorted(df["pool"].unique()):
        df[df["pool"] == pool].to_csv(out / f"{pool}_results.csv", index=False,
                                      lineterminator="\n")
    print(f"merged {len(df)} rows ({dupes} duplicates dropped) -> "
          f"{out / 'all_rows.csv'}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)


def cmd_status(_) -> None:
    load_tokens()
    out = exp_dir()
    users = active()
    n_done = sum(1 for u in users if _finished(u))
    rows = 0
    f = out / "all_rows.csv"
    if f.exists():
        rows = sum(1 for _ in open(f, encoding="utf-8")) - 1
    print(f"job={CURRENT_JOB} experiment={job_cfg()['experiment']} "
          f"pools={job_cfg()['pools']} nshards={nshards()}")
    print(f"kernels complete: {n_done}/{len(users)}   merged rows: {rows}")
    for u in users:
        print(f"  {u:<24} {_status(u)}")


def cmd_run(_) -> None:
    for step in (cmd_setup, cmd_push, cmd_poll, cmd_fetch, cmd_merge):
        t0 = time.time()
        print("=" * 70, flush=True)
        print(f"[fanout] {step.__name__} started", flush=True)
        step(None)
        print(f"[fanout] {step.__name__} done in {time.time() - t0:.0f}s",
              flush=True)


def main() -> None:
    global CURRENT_JOB
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tokens", "setup", "push", "poll", "fetch",
                                    "merge", "status", "run"])
    ap.add_argument("--job", default="main", choices=sorted(JOBS))
    ap.add_argument("--force", action="store_true",
                    help="push even when the local log already shows "
                         "EXPERIMENT_COMPLETE (re-runs a job)")
    args = ap.parse_args()
    CURRENT_JOB = args.job
    print(f"[job] {args.job}: experiment={job_cfg()['experiment']} "
          f"pools={job_cfg()['pools']} nshards={nshards()} "
          f"pin={job_cfg()['pin']} -> {slug()}")
    globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    main()
