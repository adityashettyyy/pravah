# Pravah REST API

The same closed loop as the dashboard and `pipeline.py` (GNN propagation -> RL rerouting -> before/after),
over HTTP. Built with FastAPI; interactive docs are served at `/docs`.

```bash
uvicorn api.main:app --reload            # or: make api
curl localhost:8000/health
```

The engine (GNN, RL agent, network) is loaded once per process, on first request. Calls are serialised with a lock
because the RL environment is stateful, so one process serves one simulation at a time (a single call takes about
0.2 s). Run several workers to scale out.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness, which agent is loaded (`MaskablePPO` or the rule fallback), hub counts |
| GET | `/hubs?q=&limit=` | Hubs that can be disrupted, busiest first; `q` filters by name |
| POST | `/propagation` | GNN risk spread from one hub (no rerouting) |
| POST | `/simulate` | Full loop: propagation, 10-batch rerouting plan, every policy compared |

### Request body (`/propagation`, `/simulate`)

| Field | Type | Default | Notes |
|---|---|---|---|
| `hub` | string | required | Part of a hub name. The busiest hub that matches is used |
| `severity` | float | 0.8 | 0.3 to 1.0 |
| `seed` | int | 0 | Seeds the synthetic shipment batches |
| `top_n` | int | 15 | Number of most-exposed hubs returned (1 to 100) |

```bash
curl -s -X POST localhost:8000/simulate -H 'content-type: application/json' \
     -d '{"hub": "Bhiwandi", "severity": 0.8, "seed": 0}'
```

`/simulate` returns the same JSON as `python pipeline.py --out result.json`: `propagation`, `plan` (one entry per
batch with the chosen action, delay and cost), `policies` (excess delay, cost, overloads, saving and containment for
the five policies), `latency_ms` and an `assumptions` block.

### Errors

| Status | When |
|---|---|
| 404 | No hub name contains the text |
| 422 | Invalid body (for example severity outside 0.3 to 1.0), or the matched hub has too few downstream hubs to simulate |

## What the numbers mean

Same caveats as everywhere else in the project: the cascade rule and every delay constant are assumptions, shipment
batches are synthetic, and results compare policies inside the simulation. See the README section
"What the data does and does not support".
