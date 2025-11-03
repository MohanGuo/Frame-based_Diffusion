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
from sym_nn.utils import qr, orthogonal_haar, GaussianLayer
from sym_nn.dit import DiT
EPS = 1e-8

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

        # self.egnn = EGNN_dynamics_QM9_MC(in_node_nf=in_node_nf, context_node_nf=args.context_node_nf,
        #          n_dims=3, device=device, hidden_nf=args.nf,
        #          act_fn=torch.nn.SiLU(), n_layers=3, attention=False,
        #         num_vectors=7, num_vectors_out=2
        #          )
        self.egnn = egnn

        in_node_nf -= 1
        #DiT init
        self.gaussian_embedder = GaussianLayer(K=184)
        xh_hidden_size = 184
        K = 184
        hidden_size = 384
        depth = 12
        num_heads = 6
        mlp_ratio = 4.0
        mlp_dropout = 0.0
        mlp_type = "swiglu"
        self.xh_embedder = nn.Linear(n_dims+in_node_nf+context_node_nf, xh_hidden_size)
        self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)

        self.model = DiT(
            out_channels=n_dims+in_node_nf+context_node_nf,
            hidden_size=hidden_size, depth=depth, num_heads=num_heads, 
            mlp_ratio=mlp_ratio, mlp_dropout=mlp_dropout, mlp_type=mlp_type,
            use_fused_attn=True, x_emb="identity")

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.xh_embedder.apply(_basic_init)
        self.pos_embedder.apply(_basic_init)



    def _forward(self, t, xh, node_mask, edge_mask, context):
        # start_1 = time.time()
        assert context is None
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]

        x_input =remove_mean_with_mask(xh[..., :self.n_dims], node_mask)
        x_input = x_input.view(bs*n_nodes, -1)
        
        xh = xh.view(bs*n_nodes, -1)
        h = xh[:, self.n_dims:].clone()
        
        h_egnn = h.clone()
        
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        

        ################## MG: pass egnn to get invariant coordinates ###################
        output, vel_inverse, pose = self.egnn._forward(h_egnn, x_input, edges, node_mask, edge_mask, context=None, bs=bs, n_nodes=n_nodes, dims=dims)
        x_invariant = output[..., :self.n_dims]
        x_invariant = x_invariant.view(bs, n_nodes, -1)
        x_invariant = remove_mean_with_mask(x_invariant, node_mask.view(bs, n_nodes, 1))
        #################################################################################

        ######DiT input#####
        x = xh[:, 0:self.n_dims].clone()
        x = x.view(bs, n_nodes, 3)
        node_mask = node_mask.view(bs, n_nodes, 1)
        x = remove_mean_with_mask(x, node_mask)
        N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        pos_emb = self.gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]
        pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, K]
        # xh = torch.cat([x.clone(), h.view(bs, n_nodes, -1)], dim=-1)
        xh = torch.cat([x_invariant.clone(), h.view(bs, n_nodes, -1)], dim=-1)
        xh = xh.view(bs, n_nodes, -1)
        # print(f"xh: {xh}")
        xh = self.xh_embedder(xh)
        # print(f"xh: {xh}")
        xh = node_mask * torch.cat([xh, pos_emb], dim=-1)
        # print(f"xh: {xh}")

        xh = node_mask * self.model(xh, t.squeeze(-1), node_mask.squeeze(-1))

        # print(f"xh: {xh}")
        x_final = xh[:, :, :self.n_dims] 
        h_final = xh[:, :, self.n_dims:]
        x_final = remove_mean_with_mask(x_final, node_mask)
        assert_mean_zero_with_mask(x_final, node_mask)
        x_final = x_final.view(bs * n_nodes, -1)
        h_final = h_final.view(bs * n_nodes, -1)
        
        
        ################# if transform back #####################
        # print(f"Shape of x_final: {x_final.shape}")
        # print(f"Shape of pose: {pose.shape}")
        # # #x_final: [bs*nodes, 3]
        x_final_equivariant = torch.bmm(x_final.unsqueeze(1), pose).squeeze(1)
        # vel = (x_final) * node_mask
        x_final_equivariant = x_final_equivariant.view(bs, n_nodes, -1)
        x_final_equivariant = remove_mean_with_mask(x_final_equivariant, node_mask)
        vel = (x_final_equivariant)
        # vel = x_final.view(bs, n_nodes, -1) 
        #########################################################
        # vel = x_final.view(bs, n_nodes, -1) 

        # vel = vel.view(bs, n_nodes, -1)

        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting transformer output to zero.')
            vel = torch.zeros_like(vel)

        # print(f"Shape of node_mask: {node_mask}")
        assert_mean_zero_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        # start_5 = time.time()
        # print(f'One forward took {start_5 - start_1:.2f} seconds')
        # print(f"vel: {vel}")

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