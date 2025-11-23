import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

class GRAPELoss(nn.Module):
    """
    Hybrid Loss Function:
    1. Binary Cross Entropy for Link Prediction
    2. MSE for Binding Affinity (Only on edges where affinity is measured/synthetic)
    """
    def __init__(self, lambda_affinity=0.1):
        super().__init__()
        self.alpha = lambda_affinity
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, link_pred, link_true, affinity_pred, affinity_true, affinity_mask):
        # 1. Link Prediction Loss
        loss_link = self.bce(link_pred, link_true)

        # 2. Affinity Loss (Sparse)
        mask_bool = affinity_mask.bool()
        
        if mask_bool.sum() > 0:
            # only calculate error where we have ground truth (BindingDB or High-Conf IntAct)
            loss_affinity = self.mse(affinity_pred[mask_bool], affinity_true[mask_bool])
        else:
            loss_affinity = torch.tensor(0.0, device=link_pred.device, requires_grad=True)

        return loss_link + self.alpha * loss_affinity

class EGNNLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, heads=1, dropout=0.1):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.head_dim = out_channels // heads
        self.out_channels = out_channels
        
        self.lin_query = nn.Linear(in_channels, out_channels)
        self.lin_key = nn.Linear(in_channels, out_channels)
        self.lin_value = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(edge_dim, out_channels)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * self.head_dim))
        self.lin_out = nn.Linear(out_channels, out_channels)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        nn.init.xavier_uniform_(self.lin_edge.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.lin_out.weight)

    def forward(self, x, edge_index, edge_attr):
        q = self.lin_query(x).view(-1, self.heads, self.head_dim)
        k = self.lin_key(x).view(-1, self.heads, self.head_dim)
        v = self.lin_value(x).view(-1, self.heads, self.head_dim)
        e_feat = self.lin_edge(edge_attr).view(-1, self.heads, self.head_dim)
        
        out = self.propagate(edge_index, q=q, k=k, v=v, e_feat=e_feat, size=None)
        out = out.view(-1, self.out_channels)
        out = self.lin_out(out)
        return out, edge_attr

    def message(self, q_i, k_j, e_feat, index, ptr, size_i):
        k_j_e = k_j + e_feat
        alpha = torch.cat([q_i, k_j_e], dim=-1)
        alpha = (alpha * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return k_j_e * alpha.unsqueeze(-1)

class EGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.egnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.egnn_layers.append(EGNNLayer(hidden_dim, hidden_dim, edge_dim, num_heads, dropout))
        self.dropout = nn.Dropout(dropout)
        self.pred_head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.affinity_head = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.input_proj(x))
        h = self.dropout(h)
        for layer in self.egnn_layers:
            h_new, _ = layer(h, edge_index, edge_attr)
            h = h_new + h
            h = F.relu(h)
            h = self.dropout(h)
        src, dst = edge_index
        pair_embed = torch.cat([h[src], h[dst]], dim=-1)
        return self.pred_head(pair_embed).squeeze(-1), self.affinity_head(pair_embed).squeeze(-1), h

def create_egnn_model(input_dim=128, hidden_dim=64, edge_dim=3, num_layers=2, num_heads=4, dropout=0.1, **kwargs):
    if hasattr(input_dim, 'input_dim'):
        config = input_dim
        return EGNN(
            config.input_dim, config.hidden_dim, config.edge_dim, 
            config.num_layers, config.num_heads, config.dropout
        )
    return EGNN(input_dim, hidden_dim, edge_dim, num_layers, num_heads, dropout)
