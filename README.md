# AEFS — Adaptive Explainable Feature Selection for Cross-Project Defect Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

AEFS is a four-stage feature-selection framework for Cross-Project Software
Defect Prediction (CPDP): rapid uninformative pruning, sequential forward
selection, SHAP-based refinement and cross-project stability validation.
It is evaluated on four benchmark pools (AEEEM, TeraPromise, NASA MDP,
ReLink — 80 project-versions) with Leave-One-Project-Out (LOPO)
cross-validation.

This repository is the reference implementation and the complete
experimental record: source code, benchmark data, raw result tables and
analysis scripts.

## Headline results

* Feature count reduced by **75–89%** on all four pools while staying
  competitive with the SOTA baseline BorutaSHAP (which retains 100% of the
  features): **|ΔMCC| ≤ 0.077** under identical LOPO protocols.
* **ReLink**: MCC 0.228 with 78.2% reduction. **NASA MDP**: all methods are
  near-zero MCC (defect rates of only 11–13%).
* Statistics are reported to Q1 standards: Wilcoxon signed-rank with
  **Cliff's δ** effect sizes and **Holm–Bonferroni** correction within each
  pool, **Friedman + Nemenyi** post-hoc, and explicit power discussion
  (3 of 4 pools have < 6 projects).
* Fold runtime 66.73 s on AEEEM (65× AllFeatures, 31× the next method) —
  practical for offline pre-deployment analysis.
* Seed sensitivity quantified over 5 seeds: per-target MCC std 0.033
  (AEEEM) / 0.053 (TeraPromise), against an across-target std of
  0.072 / 0.135 — the reported differences are not seed artefacts.

## Repository layout

```
AEFS_Pipeline.ipynb               main pipeline (stages 1-4, LOPO, LightGBM/RF/SVM)
AEFS_Extended_Eval.ipynb          extended evaluation (5 methods x 3 classifiers)
AEFS_ReLink_NASA_Expansion.ipynb  ReLink + NASA MDP pools
AEFS_Sensitivity_Analysis.ipynb   SHAP / stability threshold sweeps, 5-seed runs
scripts/                          pipeline + analysis + replication scripts (see below)
data/                             aeeem/ nasa/ relink/ tera/ CSVs (+ AEFS_data.zip mirror)
results/experiments/              raw per-fold rows for every experiment (sharded + merged)
results/analysis/                 descriptive stats, RQ1-RQ6 CSVs (rq1_rq2..rq6)
results/*_stage4_results.csv      per-pool x per-method summaries (LightGBM)
results/*_extended_results.csv    per-pool x per-method summaries (RF / SVM)
results/registry.json             machine-readable experiment states
sensitivity/results/              authoritative threshold sweeps, runtime, convergence
sensitivity/results_kaggle/       independent replication (unpinned, 16 workers)
sensitivity/results_pinned/       independent replication (pinned library versions)
sensitivity/results_multiseed/    5-seed runs
figures/                          matplotlib exports written by the notebooks
requirements.txt                  pinned dependencies
```

The four benchmark pools are the standard public CPDP datasets: AEEEM,
TeraPromise, NASA MDP and ReLink. `data/` holds them as per-project CSVs in
a uniform schema; `AEFS_data.zip` is an identical archive of the same files.

Manuscript sources and compiled PDFs are deliberately **not** part of this
repository; everything needed to re-run the experiments and independently
re-derive every reported number is here.

## Reproducing the results

```bash
pip install -r requirements.txt
jupyter lab                    # run the four notebooks in order, or
```

Everything below is scripted and idempotent:

```bash
python scripts/regen_sensitivity.py --pools aeeem tera   # threshold sweeps, runtime, convergence
python scripts/compute_stats.py                          # per-pool tables + Wilcoxon/Cliff's delta
python scripts/stats_q1.py                               # Holm-Bonferroni, Nemenyi, power
python scripts/analyses.py                               # RQ1-RQ5 statistics from the raw rows
python scripts/analyze_extended.py                       # extended 5-method x 3-classifier analysis
python scripts/cmp_replication.py                        # replication vs. authoritative numbers
python scripts/thread_test.py                            # thread-count nondeterminism check
```

`results/analysis/rq6.csv` is the machine-checkable link between the raw
experiment rows and the narrative claims: every claim maps onto verified
aggregate values, so a reader can re-derive the paper's numbers directly
from the CSVs in this repository.

`requirements.txt` pins the exact versions used for every reported number
in `results/` and `sensitivity/results/`.

## Independent replication

The pipeline was re-run on a second machine (Linux, 4 CPUs, Python 3.13.15,
scikit-learn 1.6.1, lightgbm 4.6.0) both with the natural library versions
and with the authoritative ones pinned. Highlights: feature-count selections
are identical on 268/272 threshold-sweep cells, and the seed-variance
statistics reproduce (0.033 / 0.054 per-target std). The three replication
runs are in `sensitivity/results_kaggle/`, `sensitivity/results_pinned/` and
`sensitivity/results_multiseed/`; `scripts/cmp_replication.py` compares them
against the authoritative tables.

A mirrored copy of the code, data and results is available as the Hugging
Face dataset [`MoshinAli/aefs-cpdp`](https://huggingface.co/datasets/MoshinAli/aefs-cpdp).

## Citing this repository

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23274376.svg)](https://doi.org/10.5281/zenodo.23274376)

Cite via [CITATION.cff](CITATION.cff) (the *Cite this repository* button
on GitHub). Release history is in [RELEASE_NOTES.md](RELEASE_NOTES.md);
the DOI above is the Zenodo **concept DOI**, which always resolves to
the latest release. Release `v1.0.0` is the frozen pre-submission
snapshot ([10.5281/zenodo.23274379](https://doi.org/10.5281/zenodo.23274379));
release `v1.1.0` carries the final author metadata and figure fixes.

## License

MIT — see [LICENSE](LICENSE).
