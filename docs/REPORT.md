# Pravah: project report

## 1. Problem
A hub disruption (flood, strike, breakdown) delays shipments at that hub and, in a connected network, may
stress the hubs downstream. Operators need (a) to know which legs and hubs are at risk, (b) to see how far
a disruption could reach, and (c) a rerouting decision that cuts delay without overloading other hubs.
Pravah builds this on the Delhivery data as a loop: **predict (GNN) -> decide (RL) -> act -> re-predict**.

## 2. Data and target
144,867 checkpoint rows, 14,817 trips, 12 Sep to 6 Oct 2018. Times and distances are cumulative within a leg, so
rows are collapsed to 26,368 legs (max per trip, source, destination). A leg is **delayed** if
`actual_time > 2.6 x osrm_time` (about the 75th percentile of the ratio; 24.7% of legs). Hubs are geocoded
from the pincode table (97% of legs); the graph has 1,594 hubs and 2,649 directed edges weighted by traffic.
Everything is split **by trip**, so legs of one trip never appear on both sides of a split.

## 3. Is propagation observable? (analysis/)
Before modelling spread we tested whether the data shows it. It does not: upstream-hub delay predicts
downstream delay no better than a random hub (-0.022 vs -0.008); the apparent trip-chain contagion
(36% vs 18% delayed) disappears after removing each hub's own delay rate (-0.003); within-hub load vs delay is
-0.035. What *is* real: hub delay rates are stable over time (r = 0.86) and regional (0.39 with graph
neighbours, 0.43 with the state). We therefore treat propagation as a **scenario model with assumed
dynamics** (graph/cascade.py), built on real traffic and real hub vulnerability, and say so everywhere.

## 4. Leg-delay model (advanced/)
**Baseline:** XGBoost on distance, OSRM time, time features, route type and one-hot top-40 hubs: PR-AUC 0.670.

**Features (advanced/features.py).** Smoothed delay-rate encodings for source, destination, exact pair, state
and source-by-route (out-of-fold, grouped by trip, so a row never sees its own label), hub traffic, in/out
degree and PageRank computed on training legs only, haversine distance and circuity, cyclic hour. Unseen hubs
fall back to the prior. A unit test shuffles labels and verifies the encoding then has AUC ~ 0.5.

**Models.** Optuna (30 trials) tunes LightGBM with 3-fold grouped CV *inside* the training set, with encodings
rebuilt in every fold. XGBoost and ExtraTrees are added to a logistic stacker trained on out-of-fold
predictions. On validation the stack beat LightGBM by < 0.002, so the simpler LightGBM is the final model.

**Calibration and uncertainty.** Platt scaling (monotone, so ranking is unchanged; isotonic on ~2k rows
created ties and cost PR-AUC). Class-conditional split conformal prediction gives sets {delayed}, {on time}
or {both = uncertain} with a 90% coverage guarantee per class. Conformalized quantile regression gives the
extra minutes with a 90% interval. SHAP explains the LightGBM member.

**Results (test, 3,981 legs).** PR-AUC 0.670 -> 0.759 (+0.090, 95% CI [0.067, 0.113], trip-level bootstrap),
ROC-AUC 0.839 -> 0.889, F1 0.607 -> 0.687. Out-of-time (train to 26 Sep, test later 7,911 legs): 0.622 -> 0.741.
Cold-start legs (a hub with < 5 training legs): 0.594 -> 0.706. Conformal coverage 90.4% (delayed) and 90.9%
(on time); 74.7% of legs get a single confident label with 87.6% accuracy. Interval coverage 91.1% (uniform
across distance quartiles), median-prediction MAE 30.4 min vs 43.3 for a constant.

**Interpretation.** The dominant feature is the pair's own delay history: delay-proneness is a stable property
of routes and hubs. The model is strong at "which legs are usually late", and says nothing about spread.

## 5. Graph models
**Hub risk (gnn/hub_risk*.py).** On 228 labelled hubs with CV over hubs, GraphSAGE (0.660) and GAT (0.671) do
not beat XGBoost on hub features (0.706). A hybrid that adds observed delay rates of neighbouring and
geographically nearest training hubs reaches 0.719; the +0.014 is inside the repeat-to-repeat noise, so we
report it as no clear gain. Message passing over weak node features cannot use neighbours' *labels*, which
is why the graph helps little here.

**Propagation GNN (gnn/propagation.py).** A GraphConv model takes edge flows and learns the assumed cascade
simulator; on unseen origin hubs MAE is 0.022 against a 0.017 noise floor, precision@5 0.99, versus XGBoost
without edges (0.34) and a hop-decay heuristic (0.82). It answers in about 0.3 ms. It is a surrogate of the
simulator, not a predictor of observed spread.

## 6. Novel extensions (novelty/)
**Regional and monsoon scenarios.** A multi-source Monte Carlo cascade on the full graph: a state flood, a
radius around a city, or a national monsoon week where each state's severity is scaled by a rainfall index.
Outputs expected hubs disrupted, traffic exposed and the share of exposure that spills outside the hit region.
The rainfall index is a coarse climatological prior unless an IMD CSV is supplied.

**Pravah Criticality Index.** Expected share of network traffic exposed when a hub is disrupted at severity 0.8,
for all 1,594 hubs (simulator and GNN agree, rank correlation 0.98). PCI correlates 0.93 with traffic, 0.55
with degree and 0.44 with betweenness. Removing the top-80 PCI hubs leaves 42.8% of shipments served versus
46.2% for betweenness and 90.5% for random removal; ordering by traffic does about as well as PCI. The honest
reading: network dependence is concentrated in a few high-traffic hubs, PCI is a convenient traffic-weighted
ranking, and it additionally flags hubs such as Mumbai Hub (PCI rank 15, betweenness rank 82).

## 7. Reinforcement learning (rl/)
Environment: a disrupted hub, 10 shipment batches (urgent/non-urgent, sized), and per batch the choice
keep / reroute via one of 4 alternate hubs / hold. A reroute changes edge flows and the propagation GNN is
queried again (closed loop). The agent also sees the day's manifest (remaining urgent volume, next batch),
which a dispatcher would know. Reward is minus delay, cost and overload penalties.

**Method.** MaskablePPO with an action mask that removes reroutes into an over-capacity alternate. Disrupted
hubs are split 60/15/25 into train/validation/test (54/14/23 hubs). Three seeds; the seed is chosen by
validation reward; the test hubs are used once. Ablation: PPO without masking. Baselines: keep route, always
hold, random, nearest-hub rule, a stronger risk-aware rule, and a one-step lookahead that queries the simulator
(an oracle reference, not deployable).

**Results (23 unseen test hubs, 300 episodes).** The selected agent cuts excess delay from 20.9 h to 13.7 h
(-7.1 h, 95% CI [-7.6, -6.6]) and far outperforms the nearest-hub rule, which is worse than doing nothing
(23.1 h). Seed variation is small (14.0 +/- 0.4 h). Masking removes capacity violations completely
(0.00 vs 0.27 overloads per episode for plain PPO) at essentially the same delay.

**What did not work, stated plainly.** The agent does **not** beat a hand-written risk-aware rule (11.8 h; the agent is
1.9 h worse, CI [1.7, 2.1]) and reaches about 75% of the gain of the one-step oracle. Giving the agent the day's
manifest did not change that. Two caveats cut both ways: the rule reads the simulator's own delay constants, so it
is partly an oracle for this environment; but it also shows that in a one-step-dominated environment a good
heuristic using the same GNN risk is hard to beat. The honest claim is therefore: RL learns most of the benefit
from the reward alone, respects the capacity constraint, and generalizes to unseen hubs, but it is not better than
a well-designed rule here. More training steps, longer-horizon coupling between batches, or a rule-initialised
(imitation) start are the obvious next steps; we did not tune the agent on test hubs.

## 8. System and verification
`app/engine.py` runs the closed loop for both the CLI and the Streamlit dashboard (map with click-to-disrupt,
propagation playback, before/after panel, regional scenarios, criticality, leg predictor with uncertainty and
SHAP, evidence tab). `tests/` covers target and split hygiene, label-shuffle leakage, cascade invariants,
RL environment masks and determinism, engine/pipeline output, conformal coverage, and the dashboard under
Streamlit's headless AppTest. Loop latency: GNN propagation ~1 ms, full 10-batch decision loop a few ms per
policy, well under the real-time target.

## 9. Limitations (read these before presenting)
1. Cascade dynamics and RL delay constants are assumptions; the data cannot validate them.
2. Single courier, 3.5 weeks, no seasonality; no real weather feed (IMD not reachable); rainfall is a prior.
3. The delay label is a ratio threshold; short legs are flagged more often.
4. Batches in the demo are synthetic; RL numbers are simulation comparisons.
5. The leg model's distance feature should be the planned routing distance if the recorded one is unknown at dispatch.
6. PCI is largely a traffic ranking; the hub-risk hybrid gain is not statistically distinguishable from zero.

## 10. Future work
Real weather (IMD) and a multi-courier dataset to test generalization; learn propagation from time-stamped
inter-hub handoffs if such data exists; multi-step planning with a learned model of the environment;
online conformal updates as new legs arrive.
