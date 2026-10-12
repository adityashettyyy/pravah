# Changelog

## 1.1.0
- Added a REST API (`api/`): `/health`, `/hubs`, `/propagation`, `/simulate`.
- Added a multi-scenario evaluation sweep (`evaluation/`) with split-tagged hubs, paired bootstrap CIs and latency percentiles.
- Dashboard: download the rerouting plan (CSV) and the full scenario (JSON).
- Added Dockerfile, docker-compose and a GitHub Actions workflow.
- Added `docs/API.md` and `docs/CASE_STUDY_COVERAGE.md`.
- Tests: 22 -> 37.

## 1.0.0
- Leg-delay models, hub-risk GNNs, cascade simulator and propagation GNN, criticality index, regional scenarios,
  masked-PPO rerouting, closed-loop engine, CLI pipeline and Streamlit dashboard.
