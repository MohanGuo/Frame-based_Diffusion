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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_molecule(x, node_mask, batch_index=0, save_path='molecule.png'):

    coords = x[batch_index].detach().cpu().numpy()
    mask = node_mask[batch_index].squeeze().detach().cpu().numpy()
    
    valid_mask = mask > 0
    valid_coords = coords[valid_mask]
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(
        valid_coords[:, 0], 
        valid_coords[:, 1], 
        valid_coords[:, 2],
        s=50,
        c='blue'
    )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Molecule from Batch {batch_index}')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Save to: {save_path}")

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
    

    def random_rotation_matrices_3d(self, batch_size):

        axis = torch.randn(batch_size, 3, device=self.device)
        axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)

        angles = torch.rand(batch_size, 1, device=self.device) * 2 * torch.pi
        
        K = torch.zeros(batch_size, 3, 3, device=self.device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        I = torch.eye(3, device=self.device).unsqueeze(0).repeat(batch_size, 1, 1)
        sin_theta = torch.sin(angles).unsqueeze(-1)
        cos_theta = (1 - torch.cos(angles)).unsqueeze(-1)
        
        R = I + sin_theta * K + cos_theta * (K @ K)
        return R

    def _forward(self, t, xh, node_mask, edge_mask, context):
        # start_1 = time.time()
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]

        x =remove_mean_with_mask(xh[..., :self.n_dims], node_mask)
        x = x.view(bs*n_nodes, -1)
        
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
        
        # # valid_atoms = node_mask.squeeze() > 0
        # # x_mean = (x * node_mask).sum(dim=0) / valid_atoms.sum()
        # # print(f"Original Mean: {x_mean}")

        # # visualize_molecule(x.view(bs, n_nodes, 3), node_mask.view(bs, n_nodes, 1), batch_index=0)
        # # rot = self.random_rotation_matrix_3d()
        # rot = self.random_rotation_matrices_3d(bs) 
        # # angle = math.pi/4  # 45度
        # # rot = torch.tensor([
        # #     [1.0, 0.0, 0.0],
        # #     [0.0, math.cos(angle), -math.sin(angle)],
        # #     [0.0, math.sin(angle), math.cos(angle)]
        # # ], device=self.device)
        # # rot = torch.tensor([
        # #     [1.0, 0.0, 0.0],
        # #     [0.0, 0.0, -1.0],
        # #     [0.0, 1.0, 0.0]
        # # ], device=self.device)
        # # rot_x_180 = torch.tensor([
        # #     [1.0, 0.0, 0.0],
        # #     [0.0, -1.0, 0.0],
        # #     [0.0, 0.0, -1.0]
        # # ], device=self.device)
        # # x_rot = torch.matmul(rot, x.T).T
        # x = x.view(bs, n_nodes, -1)
        # x_rot = torch.einsum('bij,bnj->bni', rot, x)

        # # x_rot_batch = x_rot.view(bs, n_nodes, 3)
        # # node_mask_batch = node_mask.view(bs, n_nodes, 1)
        # # # visualize_molecule(x_rot_batch, node_mask_batch, batch_index=0, save_path='molecule_rotated_1.png')
        # # x_rot_batch = remove_mean_with_mask(x_rot_batch, node_mask_batch)
        # # # visualize_molecule(x_rot_batch, node_mask_batch, batch_index=0, save_path='molecule_rotated.png')
        # # x_rot = x_rot_batch.view(bs*n_nodes, 3)
        # # # print(f"x_rot: {x_rot}")

        # x_rot = x_rot.reshape(bs * n_nodes, 3)

        #TODO no equivariance + data aug
        h_final, x_final = self.transformer(h, x, node_mask=node_mask, batch_size=bs)
        # print(f"x_final: {x_final.shape}")
        #x_final: torch.Size([864, 3])
        vel = (x_final) * node_mask

        #rotate back
        # # vel = vel.view(-1, 3)
        # vel = vel.reshape(bs, n_nodes, 3)    # [bs, n_nodes, 3]
        # rot_inverse = rot.transpose(1, 2) # [batch_size, 3, 3]
        # vel = torch.einsum('bij,bnj->bni', rot_inverse, vel)  #
        # vel = vel.reshape(bs * n_nodes, 3)



        if context is not None:
            # Slice off context size:
            h_final = h_final[:, :-self.context_node_nf]

        # print(f"Shape of h_final in stage 1: {h_final.shape}")
        if self.condition_time:
            # Slice off last dimension which represented time.
            h_final = h_final[:, :-1]

        # print(f"Shape of h_final in stage 2: {h_final.shape}")

        vel = vel.reshape(bs, n_nodes, -1)

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