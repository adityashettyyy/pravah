# Evaluation sweep

207 scenarios = 23 hubs x 3 severities [0.5, 0.8, 1.0] x 3 shipment manifests; split filter: `test`. All delays are simulated (constants in `rl/env.py` are assumptions), so this compares policies, it does not measure savings on real shipments. The RL agent trained on `train` hubs only; use `test` rows as the unbiased result.

## All policies

| policy | excess_h | excess_sd | cost | overloads_per_ep | contained_pct | saving_pct |
|---|---|---|---|---|---|---|
| Keep planned route | 21.04 | 6.13 | 0.00 | 0.00 | 0.00 | 0.00 |
| Nearest hub (greedy rule) | 22.06 | 20.97 | 3.37 | 0.00 | 26.09 | -15.57 |
| Risk-aware rule | 11.33 | 4.35 | 0.18 | 0.00 | 45.41 | 44.17 |
| RL agent (MaskablePPO) | 13.15 | 4.59 | 0.19 | 0.00 | 28.99 | 35.31 |
| Oracle one-step lookahead | 10.79 | 4.50 | 0.16 | 0.00 | 53.14 | 46.81 |


## Paired difference in excess delay (hours, negative = agent better, 95% bootstrap CI)

| comparison | mean | low | high |
|---|---|---|---|
| RL agent (MaskablePPO) vs Keep planned route | -7.88 | -8.58 | -7.16 |
| RL agent (MaskablePPO) vs Risk-aware rule | +1.82 | +1.58 | +2.07 |
| RL agent (MaskablePPO) vs Nearest hub (greedy rule) | -8.91 | -11.71 | -6.51 |

## By hub split (train = seen in RL training)

**test** (23 hubs)

| policy | excess_h | excess_sd | cost | overloads_per_ep | contained_pct | saving_pct |
|---|---|---|---|---|---|---|
| Keep planned route | 21.04 | 6.13 | 0.00 | 0.00 | 0.00 | 0.00 |
| Nearest hub (greedy rule) | 22.06 | 20.97 | 3.37 | 0.00 | 26.09 | -15.57 |
| Risk-aware rule | 11.33 | 4.35 | 0.18 | 0.00 | 45.41 | 44.17 |
| RL agent (MaskablePPO) | 13.15 | 4.59 | 0.19 | 0.00 | 28.99 | 35.31 |
| Oracle one-step lookahead | 10.79 | 4.50 | 0.16 | 0.00 | 53.14 | 46.81 |

## By severity

**severity 0.5**

| policy | excess_h | excess_sd | cost | overloads_per_ep | contained_pct | saving_pct |
|---|---|---|---|---|---|---|
| Keep planned route | 13.96 | 2.03 | 0.00 | 0.00 | 0.00 | 0.00 |
| Nearest hub (greedy rule) | 20.80 | 21.07 | 3.37 | 0.00 | 7.25 | -54.54 |
| Risk-aware rule | 9.10 | 2.64 | 0.08 | 0.00 | 21.74 | 34.60 |
| RL agent (MaskablePPO) | 10.39 | 2.67 | 0.11 | 0.00 | 7.25 | 25.24 |
| Oracle one-step lookahead | 8.67 | 2.80 | 0.07 | 0.00 | 28.99 | 37.70 |

**severity 0.8**

| policy | excess_h | excess_sd | cost | overloads_per_ep | contained_pct | saving_pct |
|---|---|---|---|---|---|---|
| Keep planned route | 21.92 | 2.88 | 0.00 | 0.00 | 0.00 | 0.00 |
| Nearest hub (greedy rule) | 22.18 | 21.03 | 3.37 | 0.00 | 28.99 | -4.47 |
| Risk-aware rule | 11.70 | 4.12 | 0.19 | 0.00 | 52.17 | 46.48 |
| RL agent (MaskablePPO) | 13.52 | 4.22 | 0.20 | 0.00 | 33.33 | 38.05 |
| Oracle one-step lookahead | 11.13 | 4.34 | 0.16 | 0.00 | 60.87 | 49.05 |

**severity 1.0**

| policy | excess_h | excess_sd | cost | overloads_per_ep | contained_pct | saving_pct |
|---|---|---|---|---|---|---|
| Keep planned route | 27.22 | 3.35 | 0.00 | 0.00 | 0.00 | 0.00 |
| Nearest hub (greedy rule) | 23.19 | 21.04 | 3.37 | 0.00 | 42.03 | 12.32 |
| Risk-aware rule | 13.19 | 4.97 | 0.26 | 0.00 | 62.32 | 51.44 |
| RL agent (MaskablePPO) | 15.54 | 5.02 | 0.26 | 0.00 | 46.38 | 42.63 |
| Oracle one-step lookahead | 12.57 | 5.17 | 0.24 | 0.00 | 69.57 | 53.69 |

## RL agent by state

| state | hubs | excess_h | saving_pct | contained_pct |
|---|---|---|---|---|
| Andhra Pradesh | 4.00 | 14.00 | 30.73 | 16.67 |
| Gujarat | 3.00 | 14.08 | 29.07 | 37.04 |
| Maharashtra | 3.00 | 14.00 | 31.57 | 22.22 |
| West Bengal | 3.00 | 17.74 | 25.75 | 7.41 |
| Kerala | 2.00 | 8.10 | 57.41 | 77.78 |
| Karnataka | 1.00 | 10.33 | 41.33 | 33.33 |
| Haryana | 1.00 | 10.53 | 38.45 | 22.22 |
| Bihar | 1.00 | 13.59 | 43.23 | 11.11 |
| Madhya Pradesh | 1.00 | 9.54 | 44.98 | 55.56 |
| Orissa | 1.00 | 16.55 | 7.66 | 0.00 |
| Tamil Nadu | 1.00 | 11.88 | 37.73 | 11.11 |
| Telangana | 1.00 | 10.13 | 54.74 | 66.67 |
| Uttar Pradesh | 1.00 | 10.26 | 47.00 | 44.44 |

## Latency (case study: real-time decision loop)

| stage | p50 ms | p95 ms | max ms | n |
|---|---|---|---|---|
| gnn_propagation | 1.6 | 2.5 | 5.2 | 69 |
| full_loop_all_policies | 101.7 | 137.4 | 194.0 | 207 |
