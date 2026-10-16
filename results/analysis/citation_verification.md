# Citation verification report

Generated: 2026-10-16  
References checked: **47** (identical bibliography in all three manuscripts)

## Summary

| Status | Count |
|---|---|
| VERIFIED | 45 |
| VERIFIED_WITH_MINOR_VARIANCE | 2 |

Sources: Crossref (DOI/title search), doi.org handle API, arXiv, plus curated records for PMLR-only venues.

## References

| Key | Year | Status | Title (verified) |
|---|---|---|---|
| arora2025 | 2025 | VERIFIED | Explainable Artificial Intelligence Techniques for Software Development Lifecycle: A Ph... |
| azzeh2026 | 2026 | VERIFIED | Cross-project software defects prediction using fuzzy embedding and deep learning |
| bal2022 | 2023 | VERIFIED | Improving Cross-Project Software Defect Prediction Method Through Transformation and Fe... |
| cao2025 | 2026 | VERIFIED_WITH_MINOR_VARIANCE | A Systematic Literature Review on Explainability for ML/DL-based Software Engineering |
| chen2025 | 2025 | VERIFIED | MBL-CPDP: A Multi-Objective Bilevel Method for Cross-Project Defect Prediction |
| dAmbros2012 | 2012 | VERIFIED | Evaluating defect prediction approaches: a benchmark and an extensive comparison |
| dai2025 | 2025 | VERIFIED | Improve cross-project just-in-time defect prediction with dynamic transfer learning |
| dlong2013 | 2013 | VERIFIED | Transfer Feature Learning with Joint Distribution Adaptation |
| featrev2025 | 2025 | VERIFIED | Feature-based Transfer Learning in Cross-Project Defect Prediction: A Systematic Review |
| gecer2025 | 2025 | VERIFIED | Explainable AI Framework for Software Defect Prediction |
| grattan2024 | 2024 | VERIFIED | The need for more informative defect prediction: A systematic literature review |
| guney25 | 2025 | VERIFIED | Active feature acquisition via explainability-driven ranking |
| gupta2026 | 2026 | VERIFIED | Explainable Machine Learning Models for Software Defect Prediction: An Empirical Study ... |
| hall2012 | 2012 | VERIFIED | A Systematic Literature Review on Fault Prediction Performance in Software Engineering |
| jiang2024 | 2024 | VERIFIED | Cross-project defect prediction via semantic and syntactic encoding |
| jureczko2010 | 2010 | VERIFIED | Towards identifying software project clusters with regard to defect prediction |
| kartika2026 | 2026 | VERIFIED | Comparative Performance Evaluation of Linear, Bagging, and Boosting Models Using Boruta... |
| khatri2023 | 2023 | VERIFIED | An effective feature selection based cross-project defect prediction model for software... |
| kumar2025 | 2025 | VERIFIED | An optimal feature selection based hybrid intelligent model for software defect prediction |
| kun2026tabf | 2026 | VERIFIED | An explainable AI framework for enhanced software defect prediction using transformer-a... |
| lazebnik2024 | 2024 | VERIFIED | An algorithm to optimize explainability using feature ensembles |
| lee2025trust | 2025 | VERIFIED | Can We Trust the Actionable Guidance from Explainable AI Techniques in Defect Prediction? |
| lei2020 | 2020 | VERIFIED | Cross-Project Software Defect Prediction Based on Feature Selection and Transfer Learning |
| ling2024 | 2024 | VERIFIED | Cross-Project Software Defect Prediction Based on Feature Selection and Knowledge Disti... |
| liu2019 | 2019 | VERIFIED | A two-phase transfer learning model for cross-project defect prediction |
| llmfs2025 | 2025 | VERIFIED | LLM-FS-Agent: A Deliberative Role-based Large Language Model Architecture for Transpare... |
| malhotra2024 | 2024 | VERIFIED | Empirical validation of feature selection techniques for cross-project defect prediction |
| nam2013 | 2013 | VERIFIED | Transfer defect learning |
| omondiagbe2024 | 2024 | VERIFIED | Improving transfer learning for software cross-project defect prediction |
| pan2011 | 2011 | VERIFIED | Domain Adaptation via Transfer Component Analysis |
| roy2026xmentor | 2026 | VERIFIED | XMENTOR: A Rank-Aware Aggregation Approach for Human-Centered Explainable AI in Just-in... |
| samat2025 | 2025 | VERIFIED | Improving the Performance of TFS with Ensemble Learning for Cross-Project Software Defe... |
| sdpmtf2024 | 2024 | VERIFIED | SDP-MTF: A Composite Transfer Learning and Feature Fusion for Cross-Project Software De... |
| shepperd2013 | 2013 | VERIFIED | Data Quality: Some Comments on the NASA Software Defect Datasets |
| shepperd2026audit | 2026 | VERIFIED | An audit of machine learning experiments on software defect prediction |
| sotto2024 | 2024 | VERIFIED | A Survey on Transfer Learning for Cross-Project Defect Prediction |
| tong2024 | 2024 | VERIFIED | MASTER: Multi-Source Transfer Weighted Ensemble Learning for Multiple Sources Cross-Pro... |
| tristage2025 | 2025 | VERIFIED | A three-stage cross-project defect prediction framework based on feature representation... |
| turhan2009 | 2009 | VERIFIED | On the relative value of cross-company and within-company data for defect prediction |
| uddin2026 | 2026 | VERIFIED | Quantum-inspired multi-objective feature selection for Cross-Project software Defect Pr... |
| wang2026 | 2026 | VERIFIED | Cross-project defect prediction based on transfer graph convolutional network |
| wu2011 | 2011 | VERIFIED | ReLink: recovering links between bugs and changes |
| xaiqual2026 | 2026 | VERIFIED | A Qualitative Study on XAI Techniques for Software Defect Prediction |
| xu2025joint | 2025 | VERIFIED | A Joint Learning Framework for Bridging Defect Prediction and Interpretation |
| yu2025 | 2026 | VERIFIED_WITH_MINOR_VARIANCE | Dynamic Feature Selection Based on Model Interpretation for Just-in-Time Software Defec... |
| zhang2024iapcp | 2024 | VERIFIED | IAPCP: An Effective Cross‐Project Defect Prediction Model via Intra‐Domain Alignment an... |
| zimmermann2007 | 2007 | VERIFIED | Predicting Defects for Eclipse |

## Flagged entries (non-VERIFIED or source divergence)

- **cao2025**: VERIFIED_WITH_MINOR_VARIANCE (source: crossref:doi) — optionally align minor field variance (year/venue spelling)
  - field checks: {'year': 'minor_variance'}
- **yu2025**: VERIFIED_WITH_MINOR_VARIANCE (source: crossref:doi) — optionally align minor field variance (year/venue spelling)
  - field checks: {'year': 'minor_variance'}

## Method

1. Claimed metadata extracted from `paper-ist/references.bib` (preferred) and the manual `\bibitem` blocks of `paper/main.tex` and `paper/main_ist.tex`.
2. Manual-vs-BibTeX field divergence checked per key.
3. Resolution: DOI -> Crossref `/works/<doi>`; no DOI -> Crossref `query.title` search (score >= 0.55 accepted); arXiv id/search for preprints; curated overrides for DOI-less venues (PMLR).
4. DOI liveness via the doi.org handle API.
5. Field comparison: title (normalized, >= 0.92), authors (surname-token match with diacritic/apostrophe folding), venue (alias + event-name matching with token-substring fallback), year (+/-1 = minor), volume, issue, pages ("9" == "9-9"), DOI (case-insensitive).
