# AEFS v1.1.0

**Final author metadata and figure polish (2026-10-10).**

Supersedes `v1.0.0` with no changes to code, data, or any reported
number — every result CSV is identical to the v1.0.0 archive. Changes
in this release:

* **Complete author list** (7 authors) with final ordering and
  affiliations, matching the submitted manuscript and `CITATION.cff`.
  (The v1.0.0 Zenodo record was minted before the author list was
  finalized and lists only the original four.)
* `results/analysis/rq6.csv` refreshed: the 782-claim audit table with
  updated manuscript line references (claim values unchanged).
* Zenodo metadata (`CITATION.cff`, `.zenodo.json`, README) now cites
  the **concept DOI** `10.5281/zenodo.23274376`, which always resolves
  to the latest release.

## Provenance

* GitHub: https://github.com/MIHMahmudEli/aefs-cpdp-v2
* Zenodo concept DOI: https://doi.org/10.5281/zenodo.23274376
* Hugging Face mirror: https://huggingface.co/datasets/MoshinAli/aefs-cpdp
* License: MIT

---

# AEFS v1.0.0

**Final experimental record — pre-IST-submission snapshot (2026-10-10).**

This is the stable, reviewer-verifiable release of the AEFS artifact:
the reference implementation, the four public benchmark pools, the raw
per-fold results of every experiment, the analysis CSVs behind every
reported number, and the independent replication runs.

## Contents

* `AEFS_*.ipynb` — the four pipeline notebooks (main pipeline, extended
  evaluation, ReLink/NASA expansion, sensitivity analysis).
* `scripts/` — pipeline (`aefs_core`, `run_experiment`, `scheduler`,
  `fanout`, `collect_results`) and analysis/replication scripts
  (`analyses`, `compute_stats`, `stats_q1`, `analyze_extended`,
  `cmp_replication`, `thread_test`, `regen_sensitivity`).
* `data/` — AEEEM, TeraPromise, NASA MDP, ReLink as per-project CSVs
  (80 project-versions), plus an identical `AEFS_data.zip` mirror.
* `results/experiments/` — raw sharded per-fold rows plus merged
  `all_rows.csv` for every experiment (main, ablation, CPDP baselines,
  multiseed, validation, NASA/ReLink sensitivity).
* `results/analysis/` — descriptive statistics and the RQ1–RQ6 CSVs;
  `rq6.csv` is the machine-checkable claim table linking raw rows to
  the narrative claims.
* `sensitivity/` — authoritative threshold sweeps, runtime and
  convergence runs, plus three independent replication sets
  (unpinned, pinned, multi-seed).
* `requirements.txt` — exact pinned versions used for every number.

## Headline results

* Feature count reduced by **75–89%** on all four pools, competitive
  with BorutaSHAP (100% of features): **|ΔMCC| ≤ 0.077** under
  identical LOPO protocols.
* ReLink: MCC 0.228 at 78.2% reduction; NASA MDP: all methods near-zero
  MCC (defect rates 11–13%).
* Q1-standard statistics: Wilcoxon–Holm with Cliff's δ, Friedman +
  Nemenyi, explicit power discussion.

## Integrity

Every number in the associated manuscript is machine-checked against
these CSVs. Manuscript sources and compiled PDFs are deliberately
**not** included in this repository; everything needed to re-run the
experiments and independently re-derive the reported results is here.

## Provenance

* GitHub: https://github.com/MIHMahmudEli/aefs-cpdp-v2
* Zenodo DOI (v1.0.0): https://doi.org/10.5281/zenodo.23274379
* Hugging Face mirror: https://huggingface.co/datasets/MoshinAli/aefs-cpdp
* License: MIT
