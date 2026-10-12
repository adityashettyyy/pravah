"""Pravah dashboard.   Run from the repo root:   streamlit run app/dashboard.py

Tabs: live disruption simulation (GNN propagation + RL rerouting), regional / monsoon scenarios,
network criticality, single-leg delay predictor (calibrated, with uncertainty and explanation), and
an evidence tab that states what is measured and what is assumed.
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.engine import (Engine, POLICY_ORDER, AGENT, CRIT_PATH, CRIT_SUMMARY, RL_METRICS, ADV_METRICS,
                        haversine_km)
from graph.cascade import BETA, DECAY, HOPS
from novelty import scenarios as scen

INK, AMBER, GREEN, SAND, PAPER, BLUE = "#1A1815", "#C4622A", "#2F6B4F", "#B8A88A", "#FAF7F2", "#2B5C8A"
POLICY_COLORS = {"Keep planned route": "#9A8F7C", "Nearest hub (greedy rule)": SAND,
                 "Risk-aware rule": "#8FA9A3", AGENT: AMBER, "Oracle one-step lookahead": "#5B5B5B"}

st.set_page_config(page_title="Pravah | delay propagation and rerouting", page_icon="🌊", layout="wide")
st.markdown(f"""
<style>
.block-container {{padding-top: 1.4rem; max-width: 1500px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}
.pv-hero {{font-size: 2.1rem; font-weight: 700; color: {INK}; line-height: 1.1;}}
.pv-sub {{color: #6b6254; margin-bottom: .6rem;}}
.pv-chip {{display:inline-block; padding:2px 10px; border-radius:12px; font-size:.78rem; margin-right:6px;
          background:#EFE8DB; color:{INK};}}
.pv-chip.ok {{background:#DCEBE2; color:{GREEN};}} .pv-chip.warn {{background:#F6E1D3; color:{AMBER};}}
div[data-testid="stMetric"] {{background:#fff; border:1px solid #E7DFD0; border-radius:10px; padding:10px 14px;}}
.pv-note {{border-left:3px solid {AMBER}; background:#FBF1EA; padding:8px 12px; border-radius:4px; font-size:.88rem;}}
</style>""", unsafe_allow_html=True)


# ====================================================================== helpers
def show(fig, **kw):
    """st.plotly_chart across Streamlit versions (width='stretch' on new ones)."""
    try:
        return st.plotly_chart(fig, width="stretch", **kw)
    except TypeError:
        return st.plotly_chart(fig, use_container_width=True, **kw)


def show_df(df, **kw):
    try:
        return st.dataframe(df, width="stretch", hide_index=True, **kw)
    except TypeError:
        return st.dataframe(df, use_container_width=True, hide_index=True, **kw)


@st.cache_resource(show_spinner="Loading GNN, RL agent and network ...")
def get_engine():
    return Engine()


@st.cache_resource(show_spinner="Loading leg-delay model ...")
def get_leg_predictor():
    try:
        from advanced.predictor import LegPredictor
        return LegPredictor()
    except Exception as e:
        return str(e)


@st.cache_data(show_spinner=False)
def load_pairs():
    from ml.problem import load_clean
    legs = load_clean()
    g = legs.groupby(["source_center", "destination_center", "source_name", "destination_name"]).agg(
        n=("is_delayed", "size"), delay_rate=("is_delayed", "mean"),
        dist=("actual_distance_to_destination", "median"), osrm=("osrm_time_min", "median"),
        ftl=("route_type", lambda s: (s == "FTL").mean())).reset_index()
    return g.sort_values("n", ascending=False)


@st.cache_data(show_spinner="Running regional scenario ...")
def cached_region(_eng, kind, runs, **kw):
    r, df = _eng.regional(kind, runs=runs, **kw)
    return {k: v for k, v in r.items() if k not in ("sev0", "p_all", "p_cascade")}, df


@st.cache_data(show_spinner="Ranking states by exposure ...")
def cached_states(_eng, severity):
    return scen.worst_states(_eng.net, severity=severity, runs=80)


@st.cache_data(show_spinner="Running the policies on 12 random manifests ...")
def cached_robust(_eng, hub, severity):
    return _eng.robustness(hub, severity, n=12)


class Canvas:
    """One map API over online tiles (Scattermap) and an offline outline (Scattergeo)."""

    def __init__(self, offline=False, height=560):
        self.fig, self.offline, self.height = go.Figure(), offline, height
        self.T = go.Scattergeo if offline else go.Scattermap

    def markers(self, lat, lon, size, color, name, text=None, custom=None, colorscale=None, cmin=0, cmax=1,
                opacity=0.9, showscale=False, cbar_title="", showlegend=False):
        mk = dict(size=size, color=color, opacity=opacity)
        if colorscale:
            mk.update(colorscale=colorscale, cmin=cmin, cmax=cmax, showscale=showscale)
            if showscale:
                mk["colorbar"] = dict(title=cbar_title, thickness=10, len=0.6)
        self.fig.add_trace(self.T(lat=list(lat), lon=list(lon), mode="markers", marker=mk, name=name,
                                  hovertext=text, hoverinfo="text" if text is not None else "skip",
                                  customdata=custom, showlegend=showlegend))

    def lines(self, segs, color, width, name, opacity=0.8, text=None, showlegend=False):
        lat, lon = [], []
        for a, b in segs:
            lat += [a[0], b[0], None]
            lon += [a[1], b[1], None]
        if lat:
            self.fig.add_trace(self.T(lat=lat, lon=lon, mode="lines", line=dict(width=width, color=color),
                                      opacity=opacity, name=name, hoverinfo="skip", showlegend=showlegend))

    def view(self, lat, lon, pad=1.0):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        c_lat, c_lon = float((lat.min() + lat.max()) / 2), float((lon.min() + lon.max()) / 2)
        span = max((lat.max() - lat.min()) * 1.5, (lon.max() - lon.min()) * 0.9, 1.5) * pad
        zoom = float(np.clip(np.log2(633.0 / span), 3.2, 9.0))
        self.zoom, self.center = zoom, (c_lat, c_lon)
        return self

    def render(self):
        lat0, lon0 = self.center
        f = self.fig
        f.update_layout(height=self.height, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                        legend=dict(orientation="h", y=0.01, x=0.01, bgcolor="rgba(255,255,255,.75)"),
                        clickmode="event+select", dragmode="pan")
        if self.offline:
            lon_span = 633.0 / 2 ** self.zoom
            lat_span = lon_span * 0.62 * np.cos(np.radians(lat0))
            f.update_geos(projection_type="mercator", showland=True, landcolor="#F1ECE3", showcountries=True,
                          countrycolor="#9A8F7C", showsubunits=True, subunitcolor="#D8CFBE", showocean=True,
                          oceancolor="#E8EEF0", lonaxis_range=[lon0 - lon_span / 2, lon0 + lon_span / 2],
                          lataxis_range=[lat0 - lat_span / 2, lat0 + lat_span / 2], resolution=50)
        else:
            f.update_layout(map=dict(style="carto-positron", center=dict(lat=lat0, lon=lon0), zoom=self.zoom))
        return f


def policy_bars(res, title="Excess delay (hours), lower is better"):
    names = list(res)
    vals = [res[n]["excess_h"] for n in names]
    f = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=[POLICY_COLORS[n] for n in names],
                         text=[f"{v:.1f} h" for v in vals], textposition="outside"))
    f.update_layout(height=250, margin=dict(l=0, r=40, t=28, b=0), title=dict(text=title, font=dict(size=13)),
                    yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, max(vals) * 1.25 + 1]),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return f


# ====================================================================== header
eng = get_engine()
leg = get_leg_predictor()
st.markdown('<div class="pv-hero">Pravah <span style="color:#C4622A">प्रवाह</span></div>', unsafe_allow_html=True)
st.markdown('<div class="pv-sub">Predict how a hub disruption spreads through the Delhivery network, '
            'then decide how to re-route around it.</div>', unsafe_allow_html=True)
chips = [("GNN propagation model", True),
         (("RL agent (MaskablePPO)" if eng.agent_is_rl else "RL agent missing, rule fallback"), eng.agent_is_rl),
         ("Leg-delay model", not isinstance(leg, str)),
         ("Criticality index", eng.crit is not None)]
st.markdown("".join(f'<span class="pv-chip {"ok" if ok else "warn"}">{"●" if ok else "○"} {t}</span>'
                    for t, ok in chips), unsafe_allow_html=True)

with st.sidebar:
    st.header("Display")
    offline = st.toggle("Offline map (no tiles)", value=False,
                        help="Use a plain outline map if the machine has no internet for map tiles.")
    st.header("About the numbers")
    st.markdown(
        "**Real:** the Delhivery hub network, traffic per route, each hub's delay history, hub coordinates.\n\n"
        "**Assumed:** how a delay spreads between hubs (the data shows no measurable cascade), and every delay "
        "constant in the rerouting simulator. Results compare *policies inside the simulation*; they are not "
        "measured savings on real shipments.")
    st.caption(f"Cascade rule: beta={BETA}, decay={DECAY}, {HOPS} hops. See the Evidence tab.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚦 Live simulation", "🌧️ Regional / monsoon", "🎯 Network criticality",
                                        "⏱️ Leg delay predictor", "📊 Evidence"])

# ====================================================================== TAB 1
with tab1:
    options = eng.hub_options()
    labels = [l for _, l in options]
    idx_of = {l: i for i, l in options}
    if "_pending_hub" in st.session_state:
        st.session_state["hub_choice"] = st.session_state.pop("_pending_hub")
    default = next((l for l in labels if "Bhiwandi" in l), labels[0])
    st.session_state.setdefault("hub_choice", default)

    c1, c2, c3, c4 = st.columns([3, 2, 1, 1])
    hub_label = c1.selectbox("Disrupted hub (or click a hub on the map)", labels, key="hub_choice")
    severity = c2.slider("Disruption severity", 0.3, 1.0, 0.8, 0.05)
    seed = c3.number_input("Batch seed", 0, 9999, 0, help="Changes the synthetic shipment batches.")
    show_plan = c4.toggle("Show RL plan", value=True)
    hub = idx_of[hub_label]

    prop = eng.propagation(hub, severity)
    cmp = eng.compare(hub, severity, int(seed))
    H, E = prop["hubs"], prop["edges"]
    base, ag = cmp["Keep planned route"], cmp[AGENT]

    k = st.columns(6)
    k[0].metric("Hubs at risk (>20%)", prop["n_at_risk"])
    k[1].metric("Network traffic exposed", f"{prop['traffic_exposed_pct']:.1f}%")
    k[2].metric("Excess delay, planned route", f"{base['excess_h']:.1f} h")
    k[3].metric("Excess delay, with agent", f"{ag['excess_h']:.1f} h", f"{-ag['saving_pct']:.0f}%", delta_color="inverse")
    k[4].metric("Contained?", "yes" if ag["contained"] else "no", f"{ag['overloads']} overloads", delta_color="off")
    k[5].metric("GNN latency", f"{prop['latency_ms']:.1f} ms", f"loop {ag['decision_ms']:.0f} ms", delta_color="off")

    left, right = st.columns([3, 2])
    with left:
        hop_view = st.slider("Propagation step (hop from the disrupted hub)", 0, HOPS, HOPS)
        play = st.button("▶ Play propagation")
        slot = st.empty()

        def build_map(h):
            cv = Canvas(offline).view(H["lat"], H["lon"])
            cv.markers(eng.net.lat, eng.net.lon, 3, "#CFC6B4", "other hubs", opacity=0.45)
            pool_idx = np.array([i for i, _ in options])
            cv.markers(eng.net.lat[pool_idx], eng.net.lon[pool_idx], 8, SAND, "can be disrupted (click)",
                       text=[eng.label(i) for i in pool_idx], custom=[[int(i)] for i in pool_idx], opacity=0.7)
            vis = H[(H["hop"] >= 0) & (H["hop"] <= h)]
            ids = set(vis["idx"])
            ed = E[E["src"].isin(ids) & E["dst"].isin(ids)]
            segs = [((eng.net.lat[a], eng.net.lon[a]), (eng.net.lat[b], eng.net.lon[b])) for a, b in zip(ed["src"], ed["dst"])]
            cv.lines(segs, "#B5543C", 1.5, "propagation edges", opacity=0.55)
            rest = vis[vis["hop"] > 0]
            cv.markers(rest["lat"], rest["lon"], 7 + 24 * rest["risk"], rest["risk"], "predicted risk",
                       text=[f"{r.hub}<br>risk {r.risk:.2f} | hop {r.hop}" for r in rest.itertuples()],
                       colorscale="YlOrRd", cmin=0, cmax=1, showscale=True, cbar_title="risk", opacity=0.92)
            cv.markers([eng.net.lat[hub]], [eng.net.lon[hub]], 20, INK, "disrupted hub",
                       text=[f"DISRUPTED: {eng.label(hub)}"], showlegend=True)
            if show_plan and h >= HOPS:
                D = (eng.net.lat[hub], eng.net.lon[hub])
                reroute, alt_ids = [], set()
                for s in ag["steps"]:
                    if s["via_idx"] is not None:
                        a, j = s["via_idx"], s["to_idx"]
                        reroute += [(D, (eng.net.lat[a], eng.net.lon[a])),
                                    ((eng.net.lat[a], eng.net.lon[a]), (eng.net.lat[j], eng.net.lon[j]))]
                        alt_ids.add(a)
                cv.lines(reroute, GREEN, 3, "RL reroutes", opacity=0.9, showlegend=True)
                if alt_ids:
                    ai = sorted(alt_ids)
                    cv.markers(eng.net.lat[ai], eng.net.lon[ai], 14, GREEN, "alternate hubs used",
                               text=[f"alternate: {eng.label(i)}" for i in ai], showlegend=True)
            return cv.render()

        if play:
            for h in range(HOPS + 1):
                slot.plotly_chart(build_map(h), key=f"frame{h}_{time.time_ns()}")
                time.sleep(0.8)
        else:
            ev = slot.plotly_chart(build_map(hop_view), key="main_map", on_select="rerun", selection_mode="points")
            try:
                pts = ev.selection.points if ev is not None else []
                if pts:
                    cd = pts[0].get("customdata")
                    new = eng.label(int(cd[0] if isinstance(cd, (list, tuple)) else cd)) if cd is not None else None
                    if new and new in idx_of and new != hub_label:
                        st.session_state["_pending_hub"] = new
                        st.rerun()
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        st.caption("Marker size and colour = GNN-predicted risk of being disrupted. Hops are reveal order along "
                   "real shipping edges; the risk values come from the GNN, not from the hop count.")

    with right:
        show(policy_bars(cmp))
        st.markdown(f"**Before / after:** keeping the planned route costs **{base['excess_h']:.1f} h** of excess delay; "
                    f"the agent's plan costs **{ag['excess_h']:.1f} h** "
                    f"(**{ag['saving_pct']:.0f}%** less) at {ag['cost']:.2f} extra cost units.")
        if not eng.agent_is_rl:
            st.warning("RL model file not found, the 'agent' row is the risk-aware rule. Run `python -m rl.train`.")
        if st.button("Test on 12 random manifests"):
            r = cached_robust(eng, hub, severity)
            f = go.Figure(go.Bar(x=r["mean"], y=r.index, orientation="h", error_x=dict(array=r["sd"]),
                                 marker_color=[POLICY_COLORS[n] for n in r.index]))
            f.update_layout(height=250, margin=dict(l=0, r=10, t=28, b=0), yaxis=dict(autorange="reversed"),
                            title=dict(text="Mean excess delay over 12 manifests (±1 sd)", font=dict(size=13)))
            show(f)
        top = H.iloc[1:].sort_values("risk", ascending=False).head(8)
        st.markdown("**Most exposed hubs**")
        show_df(top[["hub", "state", "hop", "risk"]].assign(risk=lambda d: d["risk"].round(2)))

    st.subheader("Rerouting plan, batch by batch")
    plan = pd.DataFrame(ag["steps"])
    plan["urgent"] = plan["urgent"].map({True: "urgent", False: ""})
    plan["risk at corridor hub"] = plan.apply(lambda r: f"{r.corridor_risk_before:.2f} → {r.corridor_risk_after:.2f}", axis=1)
    show_df(plan[["batch", "to", "urgent", "size", "action", "delay_h", "risk at corridor hub", "overload"]]
            .rename(columns={"to": "destination hub", "delay_h": "delay (h)"}))
    st.caption("Closed loop: after each reroute the flows change, the GNN is queried again, and the corridor "
               "hub's risk is updated (last column). The next decision sees that new state.")
    # export: hand the plan to an operations team as CSV, or the whole scenario as JSON
    steps_df = pd.DataFrame(ag["steps"])
    scenario = {"disrupted_hub": hub_label, "severity": severity, "seed": int(seed),
                "agent": ag["policy"], "hubs_at_risk": prop["n_at_risk"],
                "traffic_exposed_pct": round(prop["traffic_exposed_pct"], 2),
                "policies": {p: {"excess_delay_h": round(r["excess_h"], 2), "cost": round(r["cost"], 2),
                                 "overloads": r["overloads"], "saving_pct": round(r["saving_pct"], 1)}
                             for p, r in cmp.items()},
                "plan": ag["steps"]}
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("⬇ Plan (CSV)", steps_df.to_csv(index=False), file_name="pravah_reroute_plan.csv",
                       mime="text/csv")
    d2.download_button("⬇ Scenario (JSON)", json.dumps(scenario, indent=2, default=lambda o: o.item()), file_name="pravah_scenario.json",
                       mime="application/json")

# ====================================================================== TAB 2
with tab2:
    st.subheader("Regional and monsoon scenarios")
    idx_rain, rain_src = scen.load_rain_index()
    st.markdown(f'<div class="pv-note">Rainfall intensity per state: <b>{rain_src}</b>. The IMD feed is not bundled; '
                f'drop a CSV <code>state,rainfall_mm</code> at <code>data/raw/imd_state_rainfall.csv</code> and it is '
                f'used automatically.</div>', unsafe_allow_html=True)
    kind_label = st.radio("Scenario", ["State-wide flood / cyclone", "Radius around a city", "National monsoon week"],
                          horizontal=True)
    states = sorted({s for s in eng.net.state if sum(1 for x in eng.net.state if x == s) >= 5})
    cities = {"Mumbai": (19.08, 72.88), "Chennai": (13.08, 80.27), "Kolkata": (22.57, 88.36), "Delhi": (28.61, 77.21),
              "Bengaluru": (12.97, 77.59), "Guwahati": (26.14, 91.74), "Kochi": (9.93, 76.27)}
    if kind_label.startswith("State"):
        st_sel = st.selectbox("State", states, index=states.index("Maharashtra") if "Maharashtra" in states else 0)
        sv = st.slider("Share of the state's hubs disrupted", 0.1, 1.0, 0.6, 0.05)
        kind, kw = "state", dict(state=st_sel, severity=sv)
    elif kind_label.startswith("Radius"):
        city = st.selectbox("Centre", list(cities))
        rad = st.slider("Radius (km)", 50, 600, 250, 25)
        sv = st.slider("Peak severity at the centre", 0.2, 1.0, 0.9, 0.05)
        kind, kw = "radius", dict(lat=cities[city][0], lon=cities[city][1], radius_km=float(rad), severity=sv)
    else:
        inten = st.slider("Monsoon intensity (scales every state's rainfall index)", 0.1, 1.0, 0.5, 0.05)
        kind, kw = "monsoon", dict(intensity=inten)
    r, rdf = cached_region(eng, kind, 150, **kw)
    m = st.columns(4)
    m[0].metric("Hubs directly hit (expected)", f"{r['directly_hit_hubs']:.0f}")
    m[1].metric("Hubs disrupted incl. cascade", f"{r['expected_hubs_disrupted']:.0f}")
    m[2].metric("Network traffic exposed", f"{r['traffic_exposed_pct']:.1f}%")
    m[3].metric("Spillover outside region", f"{r['spillover_share_pct']:.0f}%")
    cv = Canvas(offline, 520).view(eng.net.lat, eng.net.lon, pad=0.75)
    cv.markers(rdf["lat"], rdf["lon"], 3, "#CFC6B4", "untouched", opacity=0.4)
    casc = rdf[rdf["p_cascade"] > 0.05]
    cv.markers(casc["lat"], casc["lon"], 6 + 14 * casc["p_cascade"], casc["p_cascade"], "cascade victims",
               text=[f"{h}<br>cascade risk {p:.2f}" for h, p in zip(casc["hub"], casc["p_cascade"])],
               colorscale="Oranges", cmin=0, cmax=1, showscale=True, cbar_title="cascade risk")
    d = rdf[rdf["direct"] > 0]
    cv.markers(d["lat"], d["lon"], 6 + 8 * d["direct"], INK, "directly hit", text=[f"{h}<br>direct severity {s:.2f}" for h, s in zip(d["hub"], d["direct"])],
               opacity=0.8, showlegend=True)
    show(cv.render())
    out = rdf[~rdf["state"].isin(r["regions"])].sort_values("p_cascade", ascending=False).head(10)
    st.markdown("**Hubs outside the hit region most likely to be dragged in**")
    show_df(out[["hub", "state", "p_cascade", "traffic"]].assign(p_cascade=lambda x: x["p_cascade"].round(2)))
    st.markdown("**State criticality: traffic exposed if the whole state is hit at severity 0.8**")
    ranking = cached_states(eng, 0.8)
    f = go.Figure(go.Bar(x=ranking["state"].head(15), y=ranking["traffic_exposed_pct"].head(15), marker_color=AMBER))
    f.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="% of network traffic exposed")
    show(f)

# ====================================================================== TAB 3
with tab3:
    st.subheader("Pravah Criticality Index (PCI)")
    if eng.crit is None:
        st.info("Run `python -m novelty.criticality` to create the criticality table.")
    else:
        crit = eng.crit
        st.markdown("PCI = expected share of **network traffic exposed** when a hub is disrupted at severity 0.8 "
                    "(cascade simulation). Degree and betweenness rank hubs by position; PCI also accounts for how "
                    "traffic and vulnerability carry a disruption downstream.")
        sm = None
        if os.path.exists(CRIT_SUMMARY):
            import json
            sm = json.load(open(CRIT_SUMMARY))
        topn = st.slider("Show top N hubs", 10, 200, 50, 10)
        t = crit.sort_values("pci", ascending=False).head(topn)
        cv = Canvas(offline, 520).view(eng.net.lat, eng.net.lon, pad=0.75)
        cv.markers(crit["lat"], crit["lon"], 3, "#CFC6B4", "other hubs", opacity=0.4)
        cv.markers(t["lat"], t["lon"], 8 + 30 * t["pci"] / t["pci"].max(), t["pci"], "top hubs",
                   text=[f"{h}<br>PCI {p:.2f}% | degree rank {int(rd)}" for h, p, rd in zip(t["hub"], t["pci"], t["rank_degree"])],
                   colorscale="YlOrRd", cmin=0, cmax=float(t["pci"].max()), showscale=True, cbar_title="PCI %")
        show(cv.render())
        a, b = st.columns([3, 2])
        with a:
            show_df(t.head(15)[["hub", "state", "pci", "exp_hubs", "rank_degree", "rank_betw"]]
                    .rename(columns={"pci": "PCI %", "exp_hubs": "expected hubs hit", "rank_degree": "degree rank",
                                     "rank_betw": "betweenness rank"}).round(2))
        with b:
            if sm:
                f = go.Figure()
                for name, v in sm["attack"].items():
                    f.add_trace(go.Scatter(x=v["k"], y=[100 * x for x in v["served"]], mode="lines+markers", name=name))
                f.update_layout(height=330, margin=dict(l=0, r=0, t=30, b=0), xaxis_title="hubs removed",
                                yaxis_title="shipments still served (%)", title=dict(text="Targeted hub failure", font=dict(size=13)))
                show(f)
                st.caption("Spearman rank correlation of PCI with: " + ", ".join(f"{k} {v:.2f}" for k, v in sm["spearman"].items())
                           + f"; GNN surrogate {sm['spearman_gnn']:.2f}")

# ====================================================================== TAB 4
with tab4:
    st.subheader("Will this leg be late?")
    if isinstance(leg, str):
        st.info("Run `python -m advanced.train` to create the leg-delay model. (" + leg[:120] + ")")
    else:
        pairs = load_pairs()
        top_pairs = pairs.head(300).reset_index(drop=True)
        labels_p = [f"{r.source_name} → {r.destination_name}  ({r.n} legs)" for r in top_pairs.itertuples()]
        c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
        pick = c1.selectbox("Hub pair (from the 300 most frequent in the data)", labels_p)
        route = c2.radio("Route type", ["Carting", "FTL"], horizontal=True)
        dow = c3.selectbox("Weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=1)
        hour = c4.slider("Departure hour", 0, 23, 9)
        pr = top_pairs.iloc[labels_p.index(pick)]
        net = eng.net
        legs = pd.DataFrame([{"source_center": pr.source_center, "destination_center": pr.destination_center,
                              "route_type": route,
                              "od_start_time": pd.Timestamp("2018-09-17") + pd.Timedelta(days=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].index(dow), hours=hour),
                              "actual_distance_to_destination": float(pr.dist), "osrm_time_min": float(pr.osrm)}])
        res = leg.predict(legs).iloc[0]
        m = st.columns(4)
        m[0].metric("Probability of delay", f"{res.p_delay:.0%}")
        m[1].metric("Verdict (90% conformal)", res.verdict)
        m[2].metric("Extra time vs routing estimate", f"{res.extra_min_med:.0f} min",
                    f"90% interval {res.extra_min_lo:.0f} to {res.extra_min_hi:.0f}", delta_color="off")
        m[3].metric("This pair's history", f"{pr.delay_rate:.0%} late", f"{int(pr.n)} legs", delta_color="off")
        st.caption("'delayed' / 'on time' appear only when the model is confident at the 90% level; otherwise "
                   "'uncertain', so an operator knows when to look closer. Delay = actual time more than 2.6× the OSRM estimate.")
        exp = leg.explain(legs)
        f = go.Figure(go.Bar(x=[e[1] for e in exp][::-1], y=[e[0] for e in exp][::-1], orientation="h",
                             marker_color=[AMBER if e[1] > 0 else GREEN for e in exp][::-1]))
        f.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0),
                        title=dict(text="Why: SHAP contribution to the delay score (orange pushes towards delay)", font=dict(size=13)))
        show(f)

# ====================================================================== TAB 5
with tab5:
    st.subheader("Evidence and honest limits")
    adv = eng.metrics_json(ADV_METRICS)
    rl = eng.metrics_json(RL_METRICS)
    a, b = st.columns(2)
    with a:
        st.markdown("**1. Leg delay model (non-graph, advanced)**")
        if adv:
            rows = adv["test"]
            show_df(pd.DataFrame(rows).T[["pr_auc", "roc_auc", "brier", "f1", "precision", "recall"]].round(3).reset_index()
                    .rename(columns={"index": "model"}))
            st.caption(f"Out-of-time PR-AUC: baseline {adv['out_of_time']['baseline']:.3f} vs advanced "
                       f"{adv['out_of_time']['advanced']:.3f}. Conformal coverage of delayed legs "
                       f"{adv['conformal']['coverage_delayed']:.0%}; confident on {adv['conformal']['confident_share']:.0%} of legs.")
        else:
            st.info("Run `python -m advanced.train`.")
    with b:
        st.markdown("**2. RL rerouting (unseen test hubs)**")
        if rl:
            t = pd.DataFrame(rl["table"]).T[["excess_h", "excess_sd", "overloads", "contained", "reward"]].round(2).reset_index()
            show_df(t.rename(columns={"index": "policy"}))
        else:
            st.info("Run `python -m rl.train`.")
    for title, path in [("Leg model comparison, calibration, SHAP", "advanced/reports/fig6_advanced.png"),
                        ("RL policies", "rl/reports/fig_policies.png"),
                        ("Network criticality", "novelty/reports/fig_criticality.png"),
                        ("Baseline model comparison", "ml/reports/fig4_model_comparison.png")]:
        p = os.path.join(ROOT, path)
        if os.path.exists(p):
            st.markdown(f"**{title}**")
            st.image(p)
    st.markdown("**3. Propagation evidence: what the data does and does not show**")
    ev = os.path.join(ROOT, "analysis", "reports", "propagation_evidence.txt")
    if os.path.exists(ev):
        st.code(open(ev).read(), language=None)
    st.markdown("""
| Piece | Status |
|---|---|
| Hub graph, coordinates, traffic, hub delay-proneness | **Measured** from Delhivery + pincode data |
| Cross-hub delay cascade | **Assumed** (the data shows none); simulator in `graph/cascade.py` |
| Propagation GNN | Surrogate of the assumed simulator; accuracy is against simulated labels |
| Rerouting delay constants | **Assumed**, in `rl/env.py` |
| Rainfall intensity | **Assumed prior** unless an IMD CSV is supplied |
| Road distance between hubs | Haversine × 1.3 where no logged leg exists (no routing API in the build environment) |
""")
