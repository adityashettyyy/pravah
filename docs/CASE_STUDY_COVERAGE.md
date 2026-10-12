# Case study coverage

How each requirement of `Pravah_Case_Study.md` maps to this repository, including what was **not** done and why.

## Objectives (section 3)

| Objective | Status | Where |
|---|---|---|
| 1. Model the hub network as a graph and predict propagation | Done, with a caveat | `graph/`, `gnn/propagation.py`. The cascade rule is assumed because `analysis/` shows no measurable cross-hub propagation in this dataset |
| 2. RL agent for rerouting, benchmarked against rule-based and random baselines | Done | `rl/`. Masked PPO beats keep / nearest-hub / random, does **not** beat the hand-written risk-aware rule |
| 3. Geospatial enrichment | Partial | Pincode geocoding done (`data/prepare_geo.py`, 97% of legs). Road distance is haversine x 1.3, rainfall is a coarse prior (see below) |
| 4. Live dashboard, demo under two minutes | Done | `app/dashboard.py` (click-to-disrupt, propagation playback, before/after, plan export) |

## Components (section 6)

| Item | Status | Notes |
|---|---|---|
| GraphSAGE / GAT in PyG | Done | `gnn/hub_risk.py`, `gnn/hub_risk_plus.py`, `gnn/propagation.py` |
| XGBoost non-graph baseline | Done | Reported next to every GNN result. GNNs do not beat XGBoost on hub risk (reported as is) |
| Gymnasium environment, PPO via SB3 | Done | `rl/env.py`, `rl/train.py` (MaskablePPO from sb3-contrib) |
| Closed loop predict -> decide -> act -> re-predict | Done | `app/engine.py`: after each reroute the GNN is queried again |
| "Single script / API" (week 9) | Done | `pipeline.py` (CLI) and `api/` (REST, see `docs/API.md`) |
| Streamlit + Plotly map | Done | Plotly map with an offline fallback |

## Evaluation plan (section 10)

| Metric | Status | Where |
|---|---|---|
| GNN precision, recall, AUC vs XGBoost | Done | `gnn/reports/*.txt` |
| RL delay, cost, % contained vs greedy and random | Done | `rl/reports/rl_report.txt` (300 episodes, held-out hubs) |
| Multiple scenarios across hubs and regions (week 11) | Done | `evaluation/sweep.py` -> `evaluation/reports/sweep_report.md` (hubs x severities x manifests, by state, split-tagged). Regional scenarios: `novelty/scenarios.py` |
| End-to-end latency | Done | `pipeline.py` reports it per run; the sweep reports p50 / p95 over all scenarios |

## Not done, and why

| Case-study item | Status | Reason / how to close it |
|---|---|---|
| OSM road network / OSRM routing | Not done | No routing service reachable at build time. `rl/` and `novelty/` use haversine x 1.3 for pairs with no logged leg. Point the distance lookup at an OSRM instance to replace it |
| IMD rainfall data | Not done | The IMD API was not reachable. `novelty/scenarios.py` uses a hand-written state prior and switches to real data automatically if `data/raw/imd_state_rainfall.csv` (`state,rainfall_mm`) exists |
| Multi-courier validation dataset (section 5.5) | Not done | Optional in the case study, and the dataset has known quality problems. Would need a separate loader |
| Weights & Biases tracking | Not done | Marked optional. Reports and metrics JSON are committed instead |
| Observed (not assumed) propagation | Not possible here | The data does not show it (`analysis/reports/propagation_evidence.txt`). Needs timestamped per-hub congestion data |
