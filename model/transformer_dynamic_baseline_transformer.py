import torch
import torch.nn as nn
import math
import numpy as np
# from .transformer import Transformer
# from .transformer_new import Transformer
from .transformer_baseline import Transformer
# from .mpnn import LEquiMPNNQM9
import sys
sys.path.append('..')
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask, assert_mean_zero_with_mask
# from egnn.models import EGNN_dynamics_QM9_MC
from egnn.egnn_mc import EGNN as EGNN_mc
import time
# from torch_geometric.nn import global_mean_pool, global_add_pool
EPS = 1e-8
def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result

class TransformerDynamics_2(nn.Module):
    def __init__(self, args,in_node_nf, context_node_nf, n_dims, 
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
            edge_dim = 5 #4
        )
        self.args = args
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
    
    def random_rotation_matrix_3d(self):
        # 随机生成欧拉角
        theta = torch.rand(1) * 2 * torch.pi  # 绕 Z 轴的旋转
        phi = torch.rand(1) * 2 * torch.pi    # 绕 Y 轴的旋转
        psi = torch.rand(1) * 2 * torch.pi    # 绕 X 轴的旋转

        # 计算绕 Z 轴的旋转矩阵
        Rz = torch.tensor([
            [torch.cos(theta), -torch.sin(theta), 0],
            [torch.sin(theta), torch.cos(theta), 0],
            [0, 0, 1]
        ], device=self.device).reshape(3, 3)

        # 计算绕 Y 轴的旋转矩阵
        Ry = torch.tensor([
            [torch.cos(phi), 0, torch.sin(phi)],
            [0, 1, 0],
            [-torch.sin(phi), 0, torch.cos(phi)]
        ], device=self.device).reshape(3, 3)

        # 计算绕 X 轴的旋转矩阵
        Rx = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(psi), -torch.sin(psi)],
            [0, torch.sin(psi), torch.cos(psi)]
        ], device=self.device).reshape(3, 3)

        # 组合旋转矩阵
        rotation_matrix = Rz @ Ry @ Rx

        return rotation_matrix

    def _forward(self, t, xh, node_mask, edge_mask, context):
        # start_1 = time.time()
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]

        x_input =remove_mean_with_mask(xh[..., :self.n_dims], node_mask)
        x_input = x_input.view(bs*n_nodes, -1)
        
        xh = xh.view(bs*n_nodes, -1)#.clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
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
        
        rot = self.random_rotation_matrix_3d()
        rot = torch.tensor([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], device=self.device, dtype=torch.float)  # 绕Z轴旋转180度
        print(f"x: {x[1]}")
        print(f"rot: {rot}")
        x_rot = torch.matmul(rot, x.T).T
        print(f"x_rot: {x_rot[1]}")


        #TODO no equivariance + data aug
        h_final, x_final = self.transformer(h, x, node_mask=node_mask, batch_size=bs)
        vel = (x_final) * node_mask


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