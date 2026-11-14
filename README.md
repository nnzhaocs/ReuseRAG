
This directory accompanies the paper **“When the Bill Falls: Reusing the Old Index Defers, Never Avoids, Re-encoding in Complete Embedding-Model Migration”

The study examines embedding-model migration with a source index still serving queries. It compares rebuilding, adapter-based reuse, window re-ranking, and candidate-union strategies. The archived experiments report served retrieval quality, document-encoding cost, window coverage, and the decisions of a pre-committed migration planner.

## 1. Directory layout


ReuseRAG_ICDE2027_Artifact/
├── README.md
├── code/                         # Source code, configurations, and execution instructions
├── experiments/
│   ├── 01_Core_1M_K50/
│   │   ├── replay/                # Core replay results and run index
│   │   ├── coverage/              # Source-index window-coverage measurements
│   │   └── sweeps/                # Static window-depth measurements
│   ├── 02_Planner_D3_K25/
│   │   ├── planner/               # Planner outputs and static inputs
│   │   ├── replay/                # K'=25 replay results
│   │   ├── sample_sweep/          # D1 corpus-sample sensitivity
│   │   └── trecdl/                # Deeply judged TREC-DL measurements
│   ├── 03_Scale_2M_4M_8M/
│   │   ├── results_s2m/
│   │   ├── results_s4m/
│   │   ├── results_full/
│   │   ├── sweeps_s2m/
│   │   ├── sweeps_s4m/
│   │   └── sweeps_full/
│   ├── 04_Zipf_Skew/
│   │   ├── results_zipf/          # Initial skew sensitivity replays
│   │   ├── results_zipf_a0.3/
│   │   ├── results_zipf_a0.6/
│   │   ├── results_zipf_a0.9/
│   │   └── results_zipf_a1.2/
│   ├── 05_P5r_Union_Pricing/
│   │   ├── results_p5r/
│   │   ├── union_price/
│   │   ├── union_price_d3/
│   │   └── union_price_xarch/
│   └── 06_Final_Logs/
│       └── logs/                 # Adapter, NQ, seed, selector, and supplementary runs



## 2. Locating results reported in the paper

| Paper result | Artifact location | Notes |
| --- | --- | --- |
| Main model pairs and strategy replays (Tables IV, VIII; Fig. 1) | `experiments/01_Core_1M_K50/replay/` | Core MS MARCO replays are at `K'=50`. The P7/P8 code-corpus runs have their own depths and corpus sizes; check each result's configuration. |
| Window coverage and cost (Table VI; Figs. 2, 4) | `experiments/01_Core_1M_K50/coverage/` and `sweeps/` | Coverage and static window measurements are distinct from replay cost-at-service. |
| Adapter comparisons (Table V; Fig. 3) | `experiments/06_Final_Logs/logs/adapters_P*.log`, `qwen_adapters_P5*.log`, `qwen_p5r.log` | Use the final logs rather than older adapter-result archives. |
| Planner inputs and decisions (Tables III, VIII) | `experiments/02_Planner_D3_K25/planner/`, `prereg_and_scores/planner/` | See `plan.csv` and `planner_check.md`. |
| Pre-committed depth-25 replays (Table IX) | `experiments/02_Planner_D3_K25/replay/` | Eight pairs were scored for the committed depth-25 test; P11 is an additional diagnostic replay. |
| D1 sampling; deeply judged TREC-DL analysis (Table VII) | `experiments/02_Planner_D3_K25/sample_sweep/`, `trecdl/` | Includes sensitivity and diagnostic measurements. |
| Natural Questions (NQ), 11 model pairs | `experiments/06_Final_Logs/logs/nq_*.log` | NQ run, planning, and checking outputs are retained as logs. |
| Seed stability (45 replay settings) | `experiments/06_Final_Logs/logs/seed_P*.log` and the core seed-0 results | Nine windowed pairs, five seeds each (seed 0 plus four additional seeds). |
| Selector hypotheses and equivalence-band sensitivity (Table X; Fig. 5) | `experiments/06_Final_Logs/logs/`, particularly `tau_curve.log` | Table X also reports later statistical summaries; not every summary has a separate CSV. |
| P0 and cross-architecture P5p/P5q final runs | `experiments/06_Final_Logs/logs/p0.log`, `xarch_P5p.log`, `xarch_P5q.log` | These supersede older versions of the corresponding results. |
| P5r bridge and mapped-window follow-up | `experiments/05_P5r_Union_Pricing/results_p5r/`, `experiments/06_Final_Logs/logs/qwen_p5r.log` | Includes the same-family embedding-model bridge check. |
| Candidate-union cost comparisons | `experiments/05_P5r_Union_Pricing/union_price*/` | These are static candidate-union prices, not measured first-arrival costs. |
| Scale study at 2M, 4M, and 8,841,823 passages (Section VII; Table XI) | `experiments/03_Scale_2M_4M_8M/` | P2f, P3f, and P9f at fixed `K'=50`. |
| Workload-skew sensitivity and 36 additional Zipf replays | `experiments/04_Zipf_Skew/` and `prereg_and_scores/zipf/` | Additional replays are separate from the earlier skew sensitivity checks. |

Tables I–III provide summaries, definitions, and planner inputs; they are not separate replay batches.


## 3. Data and interpretation

The experiments use MS MARCO passages, CodeSearchNet Python, a Natural Questions subset, and TREC-DL judgments as specified in the paper. Model checkpoints and dataset preprocessing must match the configurations distributed with the source code. Large embedding caches and historical development runs are not included in this curated results directory.

Result identifiers follow the paper: `P*` for main experiment pairs, `N*` for NQ pairs, and `P*f` for full-corpus-scale pairs. `K'` denotes the candidate-window depth. `cov(K')` is the fraction of documents covered by the relevant windows. `Cserve/N` is the fraction of documents encoded by the time evaluated queries are served; depending on the background strategy, it need not equal window-only coverage. Debt denotes retrieval quality forgone relative to the target model.

**Important:** Compare results only under the same corpus, pair, window depth, traffic setting, and seed. Static window or candidate-union coverage must not be substituted for measured replay cost. Logs may include intermediate execution messages; the scored paper numbers are the reference for selecting the relevant final run.

## 4. Code and execution

The companion `code/` directory is intended to contain the final experiment implementation, model/data configurations, dependency specifications, and run instructions. Refer to the documentation distributed with that code for environment setup and execution commands. No specific shell command is assumed here because the final code package is maintained separately from these archived results.

The included JSON/TSV results, execution logs, and pre-registration records allow the reported measurements to be inspected without rerunning expensive embedding jobs. Some reported aggregate analyses were computed separately from the execution logs, so not every number is represented by a standalone generated CSV or analysis script.

## 5. Scope of this release

This is a curated collection of the runs relevant to the submitted paper, rather than an archive of every development iteration. Older `K'=500` replays, obsolete model-encoding runs, duplicate delivery snapshots, and the 20k-document development caches were intentionally excluded. The reported experiments do not constitute a serving-under-load benchmark or a multi-hardware latency evaluation; see the paper for the full limitations.
