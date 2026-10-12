"""Draws docs/architecture.png (system diagram used in the report and README)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

INK, AMBER, GREEN, SAND, BLUE = "#1A1815", "#C4622A", "#2F6B4F", "#EFE8DB", "#2B5C8A"
fig, ax = plt.subplots(figsize=(14, 7.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 7.2); ax.axis("off")


def box(x, y, w, h, title, sub, fc, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=INK, lw=1))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=tc)
    ax.text(x + w / 2, y + h - 0.7, sub, ha="center", va="top", fontsize=8.3, color=tc, linespacing=1.35)


def arrow(x1, y1, x2, y2, c=INK, rad=0.0, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.5, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label, fontsize=8, ha="center", color=c)


ax.text(7, 7.0, "Pravah: system architecture", ha="center", fontsize=14, fontweight="bold", color=INK)
box(0.2, 4.6, 2.9, 1.9, "DATA", "Delhivery trips (14.8k)\nPincode lat/long\nOSRM time / distance\nRain prior (IMD slot)", SAND)
box(3.6, 4.6, 2.9, 1.9, "HUB GRAPH", "1,594 hubs, 2,649 edges\nreal traffic per edge\nhub delay-proneness\ngeocoded nodes", SAND)
box(7.0, 4.6, 3.0, 1.9, "PROPAGATION GNN", "GraphConv surrogate of the\ncascade simulator; takes edge\nflows, re-queried by RL\n(0.3 ms)", "#F6E1D3")
box(10.5, 4.6, 3.3, 1.9, "RL AGENT", "MaskablePPO, action mask for\ncapacity; manifest look-ahead\nkeep / reroute via 4 hubs / hold\nreward = -(delay+cost+overload)", "#DCEBE2")
box(0.2, 1.6, 2.9, 2.2, "LEG DELAY MODEL", "LightGBM+XGB+ET stack\nout-of-fold hub encodings\nPlatt calibration\nconformal sets, CQR minutes\nSHAP explanations", "#E3ECF5")
box(3.6, 1.6, 2.9, 2.2, "CRITICALITY (PCI)", "expected traffic exposed per\nhub vs degree / betweenness\ntargeted-failure robustness", "#E3ECF5")
box(7.0, 1.6, 3.0, 2.2, "REGIONAL SCENARIOS", "state flood / radius / national\nmonsoon: many sources at once\nspillover outside the region", "#E3ECF5")
box(10.5, 1.6, 3.3, 2.2, "DASHBOARD (Streamlit)", "map + click-to-disrupt\npropagation playback\nbefore / after panel\nleg predictor, evidence tab", "#FFF4D6")
arrow(3.1, 5.5, 3.6, 5.5); arrow(6.5, 5.5, 7.0, 5.5); arrow(10.0, 5.6, 10.5, 5.6, GREEN, label="risk in state")
arrow(10.5, 5.2, 10.0, 5.2, AMBER, label="new flows")
ax.text(10.25, 6.75, "closed loop: predict -> decide -> act -> re-predict", ha="center", fontsize=9, color=AMBER, fontweight="bold")
for x in (1.65, 5.05, 8.5, 12.15):
    arrow(x, 3.8 if x < 12 else 3.8, x, 3.8, INK)
arrow(5.05, 4.6, 5.05, 3.8); arrow(8.5, 4.6, 8.5, 3.8); arrow(1.65, 4.6, 1.65, 3.8); arrow(12.15, 4.6, 12.15, 3.8)
ax.text(7, 0.7, "Real: graph, traffic, hub history, coordinates.   Assumed: cascade rule, delay constants, rain prior.",
        ha="center", fontsize=9.5, color=AMBER)
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "architecture.png"), dpi=130, bbox_inches="tight")
