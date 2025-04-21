import torch
import torch.nn as nn
from egnn.egnn_new import EGNN
from egnn.egnn_mc import EGNN as EGNN_mc
import sys
sys.path.append('..')
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask
import numpy as np
from torch_geometric.nn import global_mean_pool, global_add_pool

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
        h_dims = dims - self.n_dims
        # print(f"Shape: {h.shape, x.shape, node_mask.shape}")
        h_final, x_final = self.egnn(h, x, edges, edge_attr=None, node_mask=node_mask, edge_mask=edge_mask, n_nodes=n_nodes)
        # x_final: [bs*nodes, dims, channel]
        # print(f"output of egnn: {x_final[0]}")
        vel = (x_final.permute(0, 2, 1)) #* node_mask.unsqueeze(-1)
        #vel: [bs*n_nodes, channel, dims]
        # print(f"vel: {vel.shape}")
        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan in vel')

        ############### Global Frame Computation ########################
        # First calculate the mean velocity across nodes
        # Reshape vel to combine batch dimensions
        vel_reshaped = vel.view(bs, n_nodes, 2, -1)  # [bs, n_nodes, channel, 3]
        # print(f"vel_reshaped: {vel_reshaped.shape}")

        # 创建每个分子的掩码
        mask_expanded = node_mask.view(bs, n_nodes, 1, 1).expand(-1, -1, vel_reshaped.shape[2], 3)

        # 对每个分子单独计算平均速度
        vel_sum = (vel_reshaped * mask_expanded).sum(dim=1)  # [bs, channel, 3] - 在每个分子内对节点求和
        # vel_sum = (vel_reshaped).sum(dim=1)
        node_count = mask_expanded[:, :, 0, 0].sum(dim=1, keepdim=True)  # [bs, 1] - 每个分子的有效节点数
        vel_mean = vel_sum / (node_count.view(bs, 1, 1) + EPS)  # [bs, channel, 3] - 每个分子的平均速度

        # print(f"vel_mean: {vel_mean.shape}")
        
        # Use the first channel as primary direction for Gram-Schmidt
        v1 = vel_mean[:, 0, :]  # [bs, 3]
        # Use the second channel's mean as second vector
        v2 = vel_mean[:, 1, :]  # [bs, 3]
        # print(f"v1: {v1}")
        # print(f"v2: {v2}")

        # Apply Gram-Schmidt
        global_frame = self.gram_schmidt_batch(v1, v2)  # [bs, 3, 3]
        # print(f"global_frame: {global_frame[0]}")
        
        # Expand to apply to all nodes
        batch_indices = torch.arange(bs).to(self.device).repeat_interleave(n_nodes)  # [bs*n_nodes]
        vel_q = global_frame[batch_indices]  # [bs*n_nodes, 3, 3]
        # print(f"vel_q: {vel_q}")

        
        x_invariant = torch.bmm(x.unsqueeze(1), vel_q.permute(0, 2, 1)).squeeze(1)
        # x_invariant = x

        # print(f"x_invariant: {x_invariant[0]}")
#       node_mask: torch.Size([32, 25, 1])
#       x_invariant: torch.Size([800, 3])
        vel = x_invariant * node_mask.view(bs * n_nodes, 1)

        
        # print(f"Shape of vel in stage 2: {vel.shape}")

        if context is not None:
            # Slice off context size:
            print("Context.")
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
            return torch.cat([vel, h_final], dim=2), None, vel_q

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
        # print(f"n1: {n1}, n2: {n2}, n3: {n3}")
        return torch.stack([n1, n2, n3], dim=-2)