# Viva questions and straight answers

**Why is the target 2.6 x OSRM?** OSRM counts only driving time, so a normal leg already takes about 2x the
estimate. 2.6 is roughly the 75th percentile of actual/osrm (24.7% delayed). It is one constant in `ml/problem.py`.

**How do you prevent leakage?** Split by trip; preprocessing and encodings fitted on training data only; hub
delay-rate encodings are out-of-fold; hub centrality and counts use training legs only; a test shuffles labels
and checks the encoding has AUC ~ 0.5; an out-of-time split confirms the gain (0.622 -> 0.741).

**Why not just use a GNN?** We tested: GraphSAGE 0.660 and GAT 0.671 do not beat XGBoost 0.706 on hub risk.
The data shows delay-proneness is a regional hub property, not something that spreads, so message passing has
little to add. The GNN earns its place as a fast surrogate of the cascade simulator, where graph structure and
edge flows are essential (XGBoost without edges has precision@5 0.34 vs 0.99).

**Is delay propagation real in your data?** No. We tested it and it is not observable (correlation -0.02 vs -0.01
for a random hub). We therefore model propagation as an explicit assumption, label it, and use the real
network, traffic and hub vulnerability as inputs. Saying this is a strength, not a gap.

**What does the advanced model add over XGBoost?** PR-AUC 0.670 -> 0.759 (CI [0.067, 0.113] on the gain),
calibrated probabilities, conformal "uncertain" flags with 90% coverage, minutes-late intervals, SHAP reasons.

**What is conformal prediction?** A wrapper that turns any score into prediction sets with a coverage guarantee
(here 90% per class) using a held-out calibration set. We measured 90.4% / 90.9% on test.

**Why Platt scaling and not isotonic?** Isotonic on ~2k calibration rows created ties and lowered PR-AUC from
0.765 to 0.738; Platt is monotone and keeps the ranking.

**Why did you choose LightGBM over the stack?** The stack was better by < 0.002 on validation, within noise; we
pick the simpler model by a pre-set rule.

**Why masked PPO?** Capacity is a hard constraint. Masking removes invalid actions instead of hoping a penalty
teaches the agent; in the ablation, plain PPO causes 0.27 overloads per episode and masked PPO causes none, at the same delay (13.4 vs 13.7 h).

**Why compare against a rule-based baseline?** An RL result is only meaningful against a strong simple
alternative. We include nearest-hub, a risk-aware rule and a simulator-based lookahead oracle.

**Does RL beat a simple rule?** No. Masked PPO cuts excess delay 20.9 -> 13.7 h and beats the nearest-hub rule, but a hand-written risk-aware rule reaches 11.8 h (and that rule uses the simulator's own constants). We report this openly; the value of RL here is learning from reward alone, honouring the capacity constraint and generalizing to unseen hubs.

**Are the RL savings real?** They are savings inside a simulation with assumed delay constants and synthetic
batches. They show the policy ranking, not money saved.

**What is the Criticality Index and is it new?** Expected share of network traffic exposed if a hub fails. It
tracks traffic closely (rho 0.93) and differs from degree/betweenness; the novelty is modest and we say so.

**How would you add real weather?** Place an IMD state-rainfall CSV at `data/raw/imd_state_rainfall.csv`; the
regional module uses it in place of the prior without code changes.

**What would you do next?** Multi-courier and multi-season data, real weather, learning propagation from
timestamped handoffs, online recalibration of the conformal sets.
