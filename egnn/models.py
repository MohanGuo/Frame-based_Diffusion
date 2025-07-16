import torch
import torch.nn as nn
from egnn.egnn_new import EGNN
from egnn.egnn_mc import EGNN as EGNN_mc
import sys
sys.path.append('..')
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np
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

EPS = 1e-12
class EGNN_dynamics_QM9_MC(nn.Module):
    def __init__(self, in_node_nf, context_node_nf,
                 n_dims, hidden_nf=64, device='cpu',
                 act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
                #  condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                #  inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'，
                num_vectors=7, num_vectors_out=3
                 ):
        super().__init__()
        # if mode == 'egnn_dynamics':
        self.egnn = EGNN_mc(
            in_node_nf=in_node_nf + context_node_nf, in_edge_nf=0, # or 0?
            hidden_edge_nf=hidden_nf, hidden_node_nf=hidden_nf, hidden_coord_nf=hidden_nf, 
            device=device, act_fn=act_fn,
            n_layers=n_layers, attention=attention, 
            #tanh=tanh, 
            # norm_constant=norm_constant,
            # inv_sublayers=inv_sublayers, 
            # sin_embedding=sin_embedding,
            # normalization_factor=normalization_factor,
            # aggregation_method=aggregation_method,
            node_attr=1, #If not None, add the unembedded h as node attribute
            num_vectors=num_vectors,
            update_coords=True,
            num_vectors_out=num_vectors_out
            )
        # self.egnn = equivariant_layer(hidden_features = hidden_nf)
        self.feature_embedding = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims


    def _forward(self, h, x, edges, node_mask, edge_mask, context, bs, n_nodes, dims):
        # bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims

        h_final, x_final = self.egnn(h, x, edges, edge_attr=None, node_mask=node_mask, edge_mask=edge_mask, n_nodes=n_nodes)
        
        # print(f"Shape of x_final: {x_final.shape}")
        # x_final: [nodes, dims, channel]
        # print(f"Shape of node_mask: {node_mask.shape}")

        # node_mask = node_mask.unsqueeze(-1)
        
        # vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case
        vel = (x_final) * node_mask.unsqueeze(-1)
        # vel = x_final
        # print(f"Node mask: {node_mask.unsqueeze(-1).shape}")
        # vel = remove_mean_with_mask(vel, node_mask)
        # print(f"vel:{vel}")
        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan in vel')

        # num_nodes = vel.shape[0]
        # # identity = torch.eye(3, device=vel.device).unsqueeze(0).expand(num_nodes, -1, -1)
        # identity = torch.eye(vel.shape[-1], device=vel.device).unsqueeze(0)  # [1, d, d]
        # epsilon=1e-6
        # vel = vel + identity * epsilon

        # Normalize
        # node_norms = torch.norm(vel, dim=(1,2), keepdim=True)
        # vel_normalized = vel / (node_norms + 1e-8)
        # vel = vel_normalized

        # vel = x_final
        ## Itegrate the invariant input process
        # [node, coord, channel] -> [node, channel, coord]
        # vel = vel.permute(0, 2, 1)
        ############### GramSchmidt Method 1 ########################
        # vel_q = torch.linalg.qr(vel)[0].permute(0, 2, 1)
        # # vel_q: [node, channel, coord]
        # # print(f"vel_q: {vel_q}")
        # if torch.any(torch.isnan(vel_q)):
        #     print('Warning: detected nan in vel_q')

        # # Sign Adjustment
        # inner_products = torch.sum(vel_q * vel.permute(0, 2, 1), dim=-1)  # [node, channel]
        # inner_products = torch.where(
        #     torch.abs(inner_products) < 1e-12,
        #     torch.ones_like(inner_products) * 1e-12,
        #     inner_products
        # )
        # sign_mask = torch.sign(inner_products)  # [node, channel]
        # sign_mask = sign_mask.unsqueeze(-1)  # [node, channel, 1]
        # vel_q = vel_q * sign_mask  # [node, channel, coord]
        ############### GramSchmidt Method 2 ########################
        #vel: [node, coord, channel]
        vel_q = vel.permute(0, 2, 1)
        #vel_q: [node, channel, coord]
        vel_q = self.gram_schmidt_batch(vel_q[:, 0], vel_q[:, 1])
        # print(f"vel_q 1: {vel_q}")
        #vel_q: [bs*n_nodes, 3, 3]

        # # vel_q = vel.permute(0, 2, 1)
        # # vel_q = vel_q.view(bs, n_nodes, 3, 3)
        # # vel_q = vel_q.mean(dim=1)
        # # # vel_q = vel_q[:, 1, ...]
        # # vel_q = self.gram_schmidt_batch(vel_q[:, 0], vel_q[:, 1])
        # # print(f"vel_q 1: {vel_q.shape}")
        
        # # vel_q_mean_expanded = vel_q.unsqueeze(1)  # 形状变为 [bs, 1, 3, 3]
        # # vel_q_mean_expanded = vel_q_mean_expanded.expand(bs, n_nodes, 3, 3)  # 形状变为 [bs, n_nodes, 3, 3]
        # # vel_q = vel_q_mean_expanded.reshape(bs*n_nodes, 3, 3)  # 形状变为 [bs*n_nodes, 3, 3]

        # # print(f"vel_q 2: {vel_q}")
        # if torch.any(torch.isnan(vel_q)):
        #     print('Warning: detected nan in vel_q 2')
            
        #For equivariant layer
        # h_hidden = self.feature_embedding(h)
        # batch_idx = torch.arange(bs, device=self.device).repeat_interleave(n_nodes)
        # x_final = self.egnn(h_hidden, x, edges, batch_idx)
        # vel_q = (x_final) * node_mask.unsqueeze(-1)
        # h_final = h_hidden

        ####################### inverse ########################
        identity = torch.eye(vel_q.shape[-1], device=vel_q.device).unsqueeze(0)  # [1, d, d]
        epsilon= 1e-16
        vel_q_epsilon = vel_q + identity * epsilon

        # vel_inverse = torch.inverse(vel_q_epsilon)
        vel_inverse = vel_q_epsilon.permute(0, 2, 1)

        if torch.any(torch.isnan(vel_inverse)):
            print('Warning: detected nan in vel_inverse')
        #x: [bs*n_nodes, 3]
        # vel_inverse = vel_q.permute(0,2,1)
        ############################################################
        
        # x_invariant = torch.bmm(x.unsqueeze(1), vel_inverse).squeeze(1) #torch.matmul(x, vel_inverse)
        # x_invariant = torch.bmm(vel_inverse.permute(0, 2, 1), x.unsqueeze(2)).squeeze(2)
        # x_invariant = torch.bmm(vel_q_epsilon, x.unsqueeze(-1)).squeeze(-1)
        # x_invariant = torch.bmm(vel_q, x.unsqueeze(-1)).squeeze(-1)
        x_invariant = torch.bmm(x.unsqueeze(1), vel_q.permute(0, 2, 1)).squeeze(1)
        # x_invariant = x

        # print(f"Shape of x_invariant: {x_invariant.shape}")
        # x_invariant: [n_nodes, channel]
        # vel = x_invariant.mean(dim = -1)
        # print(f"x_invariant: {x_invariant}")
        # print(f"node_mask: {node_mask.shape}")
        # print(f"x_invariant: {x_invariant.shape}")
#       node_mask: torch.Size([32, 25, 1])
#       x_invariant: torch.Size([800, 3])
        vel = x_invariant * node_mask.expand(-1, 3)
        # vel = (x_invariant - x) * node_mask
        #node_mask: torch.Size([736, 1, 1])
        #x_invariant: torch.Size([736, 3])
        # vel = x_invariant
        # is_same = torch.equal(vel, x_invariant)
        # print(f"is_same: {is_same}")
        # print(f"vel: {vel}")
        
        # print(f"Shape of vel in stage 2: {vel.shape}")

        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        vel = vel.view(bs, n_nodes, -1)

        # print(f"vel: {vel}")

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting EGNN output to zero.')
            vel = torch.zeros_like(vel)
            x_final = torch.zeros_like(x_final)

        # print(f"Shape of vel: {vel.shape}")
        # print(f"Shape of node_mask: {node_mask.shape}")
        # print(f"Shape of node_mask.view(bs, n_nodes, 1): {node_mask.view(bs, n_nodes, 1).shape}")

        if node_mask is None:
            vel = remove_mean(vel)
        else:
            # vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        if h_dims == 0:
            return vel
        else:
            h_final = h_final.view(bs, n_nodes, -1)
            return torch.cat([vel, h_final], dim=2), vel_inverse, vel_q

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
    
    def gram_schmidt_batch(self, v1, v2):
        # print(f"v1: {v1}")
        # print(f"v2: {v2}")
        n1 = v1 / (torch.norm(v1, dim=-1, keepdim=True) + EPS)
        n2_prime = v2 - (n1 * v2).sum(dim=-1, keepdim=True) * n1
        # print(f"n2_prime: {n2_prime}")
        n2 = n2_prime / (torch.norm(n2_prime, dim=-1, keepdim=True) + EPS)
        n3 = torch.cross(n1, n2, dim=-1)
        return torch.stack([n1, n2, n3], dim=-2)