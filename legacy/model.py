import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class DelayPropagationGNN(nn.Module):
    """2-layer GraphSAGE regressor: predicts next-bucket delay rate per hub
    from this bucket's node features + graph structure."""

    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        out = torch.sigmoid(self.head(h)).squeeze(-1)  # delay rate in [0,1]
        return out
