import os

ROOT = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(ROOT, "data", "processed")
LEGS_PATH = os.path.join(PROCESSED, "legs.csv")
HUBS_PATH = os.path.join(PROCESSED, "hubs.csv")
GRAPH_PATH = os.path.join(PROCESSED, "graph.pt")
