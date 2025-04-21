# from cegnn_utils import MVLinear, MVSiLU, MVLayerNorm, CEMLP
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.aggr import Aggregation
import torch
# from algebra.cliffordalgebra import CliffordAlgebra
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import knn, radius, radius_graph
from typing import Union, Tuple, List
# from equigrid_utils import UNet3D, RFNet
from torch_geometric.utils import remove_self_loops

from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np

EPS = 1e-8
class MPNNLayer(nn.Module):
    """ Message Passing Layer """
    def __init__(self, edge_features=3, hidden_features=128, act=nn.SiLU):
        super().__init__()
        # self.edge_model = RFNet(edge_features, hidden_features, hidden_features, 3, omega_0=0.1, nonlinear_class=nn.GELU, norm_class=nn.Identity, use_bias=True)
        self.edge_model = nn.Sequential(nn.Linear(edge_features, hidden_features),
                                        act(),
                                        nn.Linear(hidden_features, hidden_features))
        
        self.message_model = nn.Sequential(nn.Linear(hidden_features*5, hidden_features),
                                           act(),
                                           nn.Linear(hidden_features, hidden_features))
        self.coord_message_model = nn.Sequential(nn.Linear(hidden_features*5, hidden_features),
                                           act(),
                                           nn.Linear(hidden_features, hidden_features))
        self.update_net =  nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           act(),
                                           nn.Linear(hidden_features, hidden_features))
        self.coord_update_net = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           act(),
                                           nn.Linear(hidden_features, 3))
        self.coord_norm = nn.LayerNorm(hidden_features)
        self.message_norm = nn.LayerNorm(hidden_features)
        
    def forward(self, node_embedding, pos_hidden, node_pos, edge_index):
        # h, x_hidden, x, edge
        message, coord_message = self.message(node_embedding, pos_hidden, node_pos, edge_index)
        h, x = self.update(message, coord_message, node_embedding, node_pos, edge_index[0])
        return h, x

    def message(self, node_embedding, pos_hidden, node_pos, edge_index):
        index_i, index_j = edge_index[0], edge_index[1]
        pos_i, pos_j = node_pos[index_i], node_pos[index_j]
        node_i, node_j = node_embedding[index_i], node_embedding[index_j]
        pos_hidden_i, pos_hidden_j = pos_hidden[index_i], pos_hidden[index_j]
        # edge_attr = torch.bmm(frame[index_i], (pos_i - pos_j).unsqueeze(-1)).squeeze(-1)
        edge_attr = pos_i - pos_j
        pos_embedding = self.edge_model(edge_attr)
        node_embedding = torch.cat((node_i, node_j), dim=-1)
        message = torch.cat((node_embedding, pos_embedding, pos_hidden_i, pos_hidden_j), dim=-1)
        coord_message = self.coord_message_model(message)
        message = self.message_model(message)
        return message, coord_message

    def update(self, message, coord_message, node_embedding, node_pos, index):

        """ Update node features """
        num_messages = torch.bincount(index)
        message = global_add_pool(message, index) / num_messages.unsqueeze(-1)
        message = self.message_norm(message)
        update = self.update_net(message) + node_embedding
        coord_message = global_add_pool(coord_message, index) / num_messages.unsqueeze(-1)
        coord_message = self.coord_norm(coord_message)
        coord_update = self.coord_update_net(coord_message) 
        # coord_update = torch.bmm(frame.permute(0, 2, 1), coord_update.unsqueeze(-1)).squeeze(-1) + node_pos
        coord_update += node_pos


        return update, coord_update
    
class LEquiMPNNQM9(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, in_node_nvf=1, hidden_nvf=16, device='cpu', act_fn=nn.SiLU(), n_layers=1, attention=False,
                    norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                    sin_embedding=False, normalization_factor=100, aggregation_method='sum', dropout=0):
        super().__init__()
        self.hidden_features = hidden_nf
        # print(f"Shape of in_node_nf: {in_node_nf}, {hidden_nf}")
        self.feature_embedding = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        self.coords_embedding = nn.Sequential(
            nn.Linear(3, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        # self.pose_estimator = equivariant_layer(hidden_features=hidden_nf)
        layers = []
        for i in range(n_layers):
            layers.append(MPNNLayer(hidden_features=hidden_nf))
        self.model = nn.Sequential(*layers)
        self.atom_features = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, in_node_nf)
        )

    def forward(self, h, x, edge_index, batch_nodes):
        x = x - global_mean_pool(x, batch_nodes)[batch_nodes]
        # print(f"batch_nodes: {batch_nodes}")
        batch_size = batch_nodes.max() + 1
        # print(f"Shape of input in Equiv: {h.shape}")
        h_hidden = self.feature_embedding(h)
        # pose = self.pose_estimator(h_hidden, x, edge_index, batch_nodes) # (batch_size*nodes, 3, 3)
        # rot_x = torch.bmm(pose, x.unsqueeze(-1)).squeeze(-1)
        x_hidden = self.coords_embedding(x)
        x_process = x
        for layer in self.model:
            h_hidden, x_process = layer(h_hidden, x_hidden, x_process, edge_index)
        num_message = torch.bincount(batch_nodes)
        # print(f"Shape of h_hidden: {h_hidden.shape}")
        # h_global = global_add_pool(h_hidden, batch_nodes) / num_message.unsqueeze(-1)
        h_global = h_hidden
        # print(f"shape of h_global 1: {h_global.shape}")
        h_global = self.atom_features(h_global)
        # print(f"shape of h_global: {h_global.shape}")
        return h_global, x_process
