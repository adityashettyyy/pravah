import torch

torch.set_num_threads(1)


def load_graph():
    from paths import GRAPH_PATH
    return torch.load(GRAPH_PATH, weights_only=False)


def undirected(edge_index):
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)
