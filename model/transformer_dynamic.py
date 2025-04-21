import torch
import torch.nn as nn
import math
import numpy as np
# from .transformer import Transformer
from .transformer_new import Transformer
from .mpnn import LEquiMPNNQM9
import sys
sys.path.append('..')
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask, assert_mean_zero_with_mask
from egnn.models import EGNN_dynamics_QM9_MC
from egnn.egnn_mc import EGNN as EGNN_mc
import time
from torch_geometric.nn import global_mean_pool, global_add_pool
EPS = 1e-8
class equivariant_layer(nn.Module):
    def __init__(self, hidden_features):
        super(equivariant_layer, self).__init__()
        self.hidden_features = hidden_features
        self.message_net1 = nn.Sequential(
            nn.Linear(2 * hidden_features + 1, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )
        self.message_net2 = nn.Sequential(
            nn.Linear(2 * hidden_features + 1, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, 1),
        )
        self.p = 5

    def _initialize_weights(self):
        for m in self.message_net1:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        for m in self.message_net2:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x, pos, edge_index, batch):
        dist = (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1, keepdim=True)
        vec1, vec2 = self.message(x[edge_index[0]], x[edge_index[1]], dist, pos[edge_index[0]], pos[edge_index[1]])
        vec1_out, vec2_out = global_add_pool(vec1, edge_index[0]), global_add_pool(vec2, edge_index[0])
        return self.gram_schmidt_batch(vec1_out, vec2_out)

    def gram_schmidt_batch(self, v1, v2):
        n1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + EPS)
        n2_prime = v2 - (n1 * v2).sum(dim=-1, keepdim=True) * n1
        n2 = n2_prime / (torch.norm(n2_prime, dim=-1, keepdim=True) + EPS)
        n3 = torch.cross(n1, n2, dim=-1)
        return torch.stack([n1, n2, n3], dim=-2)

    def omega(self, dist):
        out = 1 - (self.p + 1) * (self.p + 2) / 2 * (dist / 4.5) ** self.p + self.p * (self.p + 2) * (dist / 4.5) ** (self.p + 1) - self.p * (self.p + 1) / 2 * (dist / 4.5) ** (self.p + 2)
        return out

    def message(self, x_i, x_j, dist, pos_i, pos_j):
        x_ij = torch.cat([x_i, x_j, dist], dim=-1)
        mes_1 = self.message_net1(x_ij)
        mes_2 = self.message_net2(x_ij)
        coe = self.omega(dist)
        norm_vec = (pos_i - pos_j) / (torch.norm(pos_i - pos_j, dim=-1, keepdim=True) + EPS)
        return norm_vec * coe * mes_1, norm_vec * coe * mes_2

class TransformerDynamics_2(nn.Module):
    def __init__(self, args, egnn, in_node_nf, context_node_nf, n_dims, 
                 hidden_nf=64, device='cpu', n_heads=4, 
                 n_layers=4, condition_time=True, debug=False):
                #  norm_constant=0,
                #  inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):

        super().__init__()
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

        # self.egnn = EGNN_dynamics_QM9_MC(in_node_nf=in_node_nf - 1, context_node_nf=args.context_node_nf,
        #          n_dims=3, device=device, hidden_nf=args.nf,
        #          act_fn=torch.nn.SiLU(), n_layers=3, attention=False,
        #         num_vectors=7, num_vectors_out=2
        #          )
        self.egnn = egnn
        
        # self.egnn = equivariant_layer(hidden_features = hidden_nf)
        # self.feature_embedding = nn.Sequential(
        #     nn.Linear(in_node_nf - 1, hidden_nf),
        #     nn.SiLU(),
        #     nn.Linear(hidden_nf, hidden_nf)
        # )
        self.feature_embedding = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 2)
        )
        

        self.transformer = Transformer(
            args=args,
            in_node_nf=in_node_nf + context_node_nf,
            device=device,
            d_model=hidden_nf,
            num_heads=n_heads,
            num_layers=n_layers,
            d_ff=hidden_nf*4,
            dropout=0.1,
            edge_dim = 8 #4
        )
        # self.transformers = nn.ModuleList()
        # for _ in range(1):
        #     transformer = Transformer(
        #         args=args,
        #         in_node_nf=in_node_nf + context_node_nf,
        #         device=device,
        #         d_model=hidden_nf,
        #         num_heads=n_heads,
        #         num_layers=n_layers,  #presetting
        #         d_ff=hidden_nf*4,
        #         dropout=0.1,
        #         edge_dim=4
        #     )
        #     self.transformers.append(transformer)

        # self.equi_model = LEquiMPNNQM9(
        #         in_node_nf=in_node_nf + context_node_nf,
        #         in_edge_nf=1,
        #         hidden_nf=hidden_nf,
        #         device=device,
        #         # act_fn=act_fn,
        #         n_layers=n_layers)
        #         # attention=attention,
        #         # tanh=tanh)
        
        # self.debug=debug


    def _forward(self, t, xh, node_mask, edge_mask, context):
        # start_1 = time.time()
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]

        x_input =remove_mean_with_mask(xh[..., :self.n_dims], node_mask)
        x_input = x_input.view(bs*n_nodes, -1)
        
        xh = xh.view(bs*n_nodes, -1)#.clone() * node_mask
        # x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()
        
        h_egnn = h.clone()
        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h_egnn = torch.cat([h, context], dim=1)
        
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        

        ################## MG: pass egnn to get invariant coordinates ###################
        # # with torch.no_grad():
        # # print(f"Shape of h_egnn: {h_egnn.shape}")
        # # h_egnn:[864, 5]
        # h_hidden = self.feature_embedding(h_egnn)
        # batch_idx = torch.arange(bs, device=self.device).repeat_interleave(n_nodes)
        # pose = self.egnn(h_hidden, x, edges, batch_idx)
        # pose = (pose) * node_mask.unsqueeze(-1)
        # x_invariant = torch.bmm(x.unsqueeze(1), pose.permute(0, 2, 1)).squeeze(1)
        # x_invariant = x_invariant * node_mask.expand(-1, 3)
        # x_invariant = x_invariant.view(bs, n_nodes, -1)
        # if node_mask is None:
        #     x_invariant = remove_mean(x_invariant)
        # else:
        #     # vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))
        #     x_invariant = remove_mean_with_mask(x_invariant, node_mask.view(bs, n_nodes, 1))

        
        # # print(f"output in TransformerDynamics_2: {output[..., :self.n_dims]}")
        # # x_invariant = output[..., :self.n_dims]
        # x_invariant = x_invariant.view(bs*n_nodes, -1)
        # # print(f"x_invariant in TransformerDynamics_2: {x_invariant}")
        # # egnn_f: (nodes, n_dims, channels)

        ##Multichannel
        output, vel_inverse, pose = self.egnn._forward(h_egnn, x_input, edges, node_mask, edge_mask, context=None, bs=bs, n_nodes=n_nodes, dims=dims)
        x_invariant = output[..., :self.n_dims]
        x_invariant = x_invariant.view(bs*n_nodes, -1)
        #################################################################################

        if self.condition_time:
            if np.prod(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t.view(bs, 1).repeat(1, n_nodes)
                h_time = h_time.view(bs * n_nodes, 1)
            h = torch.cat([h, h_time], dim=1)
        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)

        radial, coord_diff = coord2diff(x_input, edges)
        coord_diff = coord_diff.view(bs, n_nodes, n_nodes, 3)
        radial = radial.view(bs, n_nodes, n_nodes, 1)
        pose_edge = pose.view(bs, n_nodes, 3, 3)
        coord_diff_reshaped = coord_diff.unsqueeze(-2)
        #local frame
        transformed_coord = torch.einsum('bijpk,bikm->bijpm', coord_diff_reshaped, pose_edge.permute(0, 1, 3, 2)).squeeze(-2)

        #global frame
        # pose_avg = pose_edge.mean(dim=1)
        # transformed_coord = torch.einsum('bijpk,bkm->bijpm', coord_diff_reshaped, pose_avg.permute(0, 2, 1)).squeeze(-2)

        #add h into edge features
        h_embedded = self.feature_embedding(h)
        rows, cols = edges
        h_i = h_embedded[rows]  # 源节点特征
        h_j = h_embedded[cols]  # 目标节点特征

        # 拼接特征
        node_edge_features = torch.cat([h_i, h_j], dim=1)  # 假设拼接维度是1

        # 调整形状与radial和transformed_coord匹配
        node_features_matrix = node_edge_features.view(bs, n_nodes, n_nodes, -1)

        #Normalize radial:
        #TODO
        # radial_mean = radial.mean()
        # radial_std = radial.std() + 1e-8
        # radial_normalized = (radial - radial_mean) / radial_std

        # edge_features = torch.cat([radial, transformed_coord], dim=-1)
        edge_features = torch.cat([radial, transformed_coord, node_features_matrix], dim=-1)
        # edge_features = torch.cat([radial, radial, radial, radial], dim=-1)
        # print(f"edge_features: {edge_features[0, 0, 1]}")
        

        # edge_features = None
        # if torch.any(torch.isnan(x_invariant)):
        #     print('Warning: detected nan in x_invariant')
        # if torch.any(torch.isnan(h)):
        #     print('Warning: detected nan in h')
        if torch.any(torch.isnan(edge_features)):
            print('Warning: detected nan in edge_features')
            edge_features = torch.zeros_like(edge_features)
        h_final, x_final = self.transformer(h, x_invariant, edge_features, node_mask=node_mask, batch_size=bs)
        # x_final = x_invariant
        # h_final = h
        # for transformer in self.transformers:
        #     x_equi = torch.bmm(x_final.unsqueeze(1), pose).squeeze(1)
        #     radial, coord_diff = coord2diff(x_equi, edges)
        #     coord_diff = coord_diff.view(bs, n_nodes, n_nodes, 3)
        #     radial = radial.view(bs, n_nodes, n_nodes, 1)
        #     pose_edge = pose.view(bs, n_nodes, 3, 3)
        #     coord_diff_reshaped = coord_diff.unsqueeze(-2)
        #     pose_avg = pose_edge.mean(dim=1)
        #     transformed_coord = torch.einsum('bijpk,bkm->bijpm', coord_diff_reshaped, pose_avg.permute(0, 2, 1)).squeeze(-2)
        #     edge_features = torch.cat([radial, transformed_coord], dim=-1)
        #     # edge_features = torch.cat([radial, radial, radial, radial], dim=-1)

        #     # h_final, x_final = transformer(h_final, x_final, edge_features, node_mask=node_mask, batch_size=bs, debug=self.debug)
        #     h_final, x_final = transformer(h_final, x_final, edge_features, node_mask=node_mask, batch_size=bs)

        ##MPNN
        # batch_idx = torch.arange(bs, device=self.device).repeat_interleave(n_nodes)
        # h_final, x_final = self.equi_model(h, x_invariant, edges, batch_idx)

        # vel = (x_final - x_invariant) * node_mask
        
        ################# if transform back #####################
        # # # # print(f"Shape of x_final: {x_final.shape}")
        # # #x_final: [bs*nodes, 3]
        x_final_equivariant = torch.bmm(x_final.unsqueeze(1), pose).squeeze(1)
        # # x_final_equivariant = torch.bmm(x_final.unsqueeze(1), vel_inverse.permute(0, 2, 1)).squeeze(1)
        # # # x_final_equivariant = torch.bmm(egnn_f.permute(0,2,1), x_final.unsqueeze(-1)).squeeze(-1)
        # # # x_final_equivariant = torch.bmm(egnn_f, x_final.unsqueeze(2)).squeeze(2)
        vel = (x_final_equivariant) * node_mask
        # # # # print(f"h_final: {h_final}")
        # # # print(f"x_final_equivariant: {x_final_equivariant}")

        # vel = (x_final) * node_mask
        #########################################################


        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        # print(f"Shape of h_final in stage 1: {h_final.shape}")
        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        # print(f"Shape of h_final in stage 2: {h_final.shape}")

        vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting transformer output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
            print("No mask!")
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))
            # print(f"Remove mean with mask!")

        # print(f"Shape of node_mask: {node_mask}")
        assert_mean_zero_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        # start_5 = time.time()
        # print(f'One forward took {start_5 - start_1:.2f} seconds')

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            # print(f"Shape of h_final in stage 3: {h_final.shape}")
            return torch.cat([vel, h_final], dim=2)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self._edges_dict:
            edges_dic_b = self._edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                # get edges for a single sample
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
                edges = [torch.LongTensor(rows).to(device),
                         torch.LongTensor(cols).to(device)]
                edges_dic_b[batch_size] = edges
                return edges
        else:
            self._edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff