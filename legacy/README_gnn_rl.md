# Pravah: graph, GNN and RL extension

This extends the baseline in `README.md` (delay prediction, XGBoost, no
graph). It covers the other four rows of the case study's component table.
Nothing here is tuned for a leaderboard number; the goal was to build each
piece for real, run it end to end on your actual data, and report what it
actually does - including where it doesn't work well yet.

## Status against the case study

| Component | Status |
|---|---|
| Delay prediction, XGBoost non-graph baseline | Done (see `README.md`) |
| Hub graph, pincode lat/long, road distances | Done, with caveats below |
| GNN and disruption injection | Done, weak signal, reported honestly |
| RL environment and PPO | Done, PPO underperforms a baseline, root cause identified |
| GNN -> RL integration | Done - the RL env's cost model IS the GNN's output, not a separate simulator |

Rough completion: this gets the ML side to somewhere around 55-65% of the
case study, in the sense that every component now has real, running code
and an honest evaluation - not that every component works well. Two of the
four new pieces (the GNN's next-bucket prediction, and PPO's routing) do
**not** beat their baselines yet. That's a legitimate, defensible state to
present at a review as long as you can explain *why* - which is the point
of the reports below.

## Setup

```bash
pip install torch torch_geometric gymnasium stable-baselines3
curl -sL -o data/pincode_raw.csv https://raw.githubusercontent.com/sanand0/pincode/master/data/IN.csv
python3 graph/build_graph.py      # graph/nodes.csv, graph/edges.csv, graph/graph_report.txt
python3 gnn/labels.py             # gnn/hub_bucket_labels.csv
python3 gnn/train.py              # gnn/gnn_model.pt, gnn/gnn_report.txt
python3 gnn/disruption.py         # gnn/disruption_report.txt
python3 rl/train_ppo.py           # rl/ppo_reroute.zip, rl/rl_report.txt
```
Run in that order - each stage reads the previous stage's output file.

## 1. Hub graph and coordinates (`graph/`)

Nodes = 1,657 hubs from the leg data. Coordinates come from the 6-digit PIN
code embedded in each hub code (e.g. `IND388121AAA` -> `388121`), matched
against an open PIN->lat/long table. 51% match exactly; most of the rest
fall back to the mean coordinate of hubs sharing the same first 3 digits
(same postal sorting district), a few to the first 2 digits (postal
circle). 97.5% of hubs get a coordinate; 42 don't and are dropped from the
graph. See `graph/graph_report.txt` for exact counts.

**Road distance**: there's no routing API reachable from this environment
(the sandbox's network allowlist has no OSRM/routing host), so there's no
way to get real road distance for hub pairs that never appear as a leg.
Edges that *do* appear in the data use the real OSRM-derived distance
from the raw file. Straight-line (haversine) distance is used only as the
RL environment's travel-cost proxy for hops that aren't a real logged leg,
and is labeled as such everywhere it's used - it is never presented as
road distance. Correlation between real distance and haversine on the
2,593 edges that have both: **0.61** - moderate, expected given the
underlying `actual_distance_to_destination` column is itself a coarse
aggregate (see the case study review's notes on that column).

## 2. GNN and disruption injection (`gnn/`)

**Ground truth**: the data has no "delay propagated from hub A to hub B"
label, so (per the case study review's suggestion) the target is hub x
time-bucket delay rate: split the 23 usable days into 4 buckets (~5-6 days
each - daily buckets would give most hubs 0-1 legs/day, too sparse), and
predict a hub's next-bucket delay rate from this bucket's network state.
946 hubs have enough volume (>=8 legs total) to score.

**Result**: the GNN (2-layer GraphSAGE) does **not** beat a persistence
baseline (predict = this bucket's value) on the held-out transition - MAE
0.163 vs 0.137. It clearly beats a global-mean baseline (0.163 vs 0.280),
so it's learning *something* about individual hubs, just not enough to add
value over "assume nothing changes." This is consistent with the baseline
model's own finding that hub identity is the strongest signal in this
dataset - propagation between hubs may be real but is apparently smaller
than hub-level noise at this data volume. With only 3 time transitions
total (4 buckets), this is one held-out data point, not a reliable
estimate either way. Full numbers: `gnn/gnn_report.txt`.

**Disruption injection**: spiking a hub's delay feature to 1.0 and
re-running the model shows 1-hop neighbours move ~3.2x more than 2-hop
neighbours, and non-neighbours (>2 hops away) show exactly zero change -
which is mathematically expected for a 2-layer GNN, not a sign of
anything broken. So the model does use the graph structure, correctly
bounded by its receptive field; whether the *size* of the effect reflects
real logistics behaviour is a separate, open question given the model
didn't beat persistence. Full numbers: `gnn/disruption_report.txt`.

## 3. RL environment and PPO (`rl/`)

**Environment**: an agent starts at a hub, needs to reach a destination
hub, and at each step picks which of up to 6 graph-neighbours to hop to.
One hub near the route is disrupted each episode (same injection mechanism
as `gnn/disruption.py`). Reward = -(GNN-predicted delay probability at the
hub you hop to) - (distance cost) - (a penalty for hopping without getting
closer to the destination), + a bonus for arriving, - a penalty for timing
out. Origin/destination pairs are pre-filtered by BFS to ones reachable
within the step budget on the actual (top-6-truncated) hub graph - sampling
uniformly over all 1,657 hubs made the task nearly unsolvable in early
testing (destination reach rate ~0%).

**This is the GNN -> RL integration**: the environment's delay cost comes
directly from the trained GNN's forward pass on the live (disrupted)
network state, not a separately hand-tuned simulator.

**Result**: PPO does not beat a distance-only greedy baseline (mean reward
-10.4 vs -8.6) and reaches the destination far less often (1.7% vs 19.0%
over 300 eval episodes). This was flat across three training budgets
(40k/80k/150k steps) with reward-shaping fixes in between, which rules out
"just needs more training." The likely cause: in this action space, action
slot 0-5 maps to a *different* physical hub depending on which hub the
agent is currently standing at (slots are ranked by local traffic, not a
fixed direction), so a plain feedforward policy has to relearn the
action->hub mapping at every state - a much harder credit-assignment
problem than fixed-direction actions (e.g. grid-world up/down/left/right).
The greedy baseline avoids this because it recomputes the best move
directly from the observation each step instead of learning a fixed
mapping. Next step: feed the GNN's node embeddings into the policy so it
reasons about hub identity rather than a slot index. Full numbers:
`rl/rl_report.txt`.

## What to say if your guide asks

- Why 4 time buckets, not daily? Median legs/hub over 23 days is ~11; daily
  buckets give ~0.5 legs/hub/day, too sparse for a rate. 4 buckets trades
  temporal resolution for a usable ~7 legs/hub/bucket.
- Why doesn't the GNN beat persistence? Only 3 time transitions to learn
  from, and the baseline model already showed hub identity dominates the
  delay signal in this data - there may just not be much propagation signal
  left over once you account for each hub's own baseline rate, at this data
  volume.
- Why doesn't PPO beat greedy? Contextual action semantics (explained
  above) - a known hard case for vanilla MLP policies over graph-structured
  action spaces, not a training or tuning failure.
- Why haversine and not real road distance for the RL env? No routing API
  is reachable from the dev sandbox; documented as an approximation
  everywhere it's used, with a 0.61 correlation to real distance as a
  sanity check.

None of this is a stopping point - it's a documented, reasoned state you
can build on: more time buckets (needs more data than the 3.5-week Kaggle
file has), a graph-aware policy for the RL side, and an OSRM instance (or
a real routing API key) for true road distances are the concrete next
steps, in that order of likely impact.
