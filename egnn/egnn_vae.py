import torch
import torch.nn as nn
import math
import numpy as np
from .decoder import VAE_Point
import sys
sys.path.append('..')
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask, assert_mean_zero_with_mask
from .models import EGNN_dynamics_QM9_MC
from .egnn_mc import EGNN as EGNN_mc
import time
from torch_geometric.nn import global_mean_pool, global_add_pool
from types import SimpleNamespace
EPS = 1e-8

class EGNN_VAE(nn.Module):
    def __init__(self, args, egnn, in_node_nf, context_node_nf, n_dims, 
                 hidden_nf=64, device='cpu', n_heads=4, 
                 n_layers=4, condition_time=False, debug=False):
                #  norm_constant=0,
                #  inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):

        super().__init__()
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

        self.egnn = egnn
        
        self.feature_embedding = nn.Sequential(
            nn.Linear(in_node_nf, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, 2)
        )
        
        self.vae = VAE_Point(latent_dim=8, in_node_nf=in_node_nf)


        

    def _forward(self, xh, node_mask, edge_mask, context, use_mean=False):
        # start_1 = time.time()
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]

        x_input =remove_mean_with_mask(xh[..., :self.n_dims], node_mask)
        x_input = x_input.view(bs*n_nodes, -1)
        
        xh = xh.view(bs*n_nodes, -1)#.clone() * node_mask
        x = xh[:, 0:self.n_dims].clone().view(bs, n_nodes, -1)
        x = remove_mean_with_mask(x, node_mask).view(bs*n_nodes, -1)
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
        ##Multichannel
        output, vel_inverse, pose = self.egnn._forward(h_egnn, x_input, edges, node_mask, edge_mask, context=None, bs=bs, n_nodes=n_nodes, dims=dims)
        x_invariant = output[..., :self.n_dims]
        x_invariant = x_invariant.view(bs*n_nodes, -1)
        #################################################################################

        # if self.condition_time:
        #     if np.prod(t.size()) == 1:
        #         # t is the same for all elements in batch.
        #         h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
        #     else:
        #         # t is different over the batch dimension.
        #         h_time = t.view(bs, 1).repeat(1, n_nodes)
        #         h_time = h_time.view(bs * n_nodes, 1)
        #     h = torch.cat([h, h_time], dim=1)
        if context is not None:
            # We're conditioning, awesome!
            context = context.view(bs*n_nodes, self.context_node_nf)
            h = torch.cat([h, context], dim=1)


        batch_dict = {
        'x': x_invariant,
        'h': h,
        'node_mask': node_mask.reshape(bs, n_nodes).bool(),
        'num_atoms': torch.ones(bs, dtype=torch.long, device=self.device) * n_nodes,
        'batch': torch.arange(bs, device=self.device).repeat_interleave(n_nodes)
        }
        batch_obj = SimpleNamespace(**batch_dict)
        out, encoded_batch = self.vae(batch_obj, use_mean=use_mean)
        
        x_final = out['x']  # [bs*n_nodes, 3]
        h_final = out['h']  # [bs*n_nodes, h_dim]
        # print(f"x_final: {x_final.shape}")    
        
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
        # if self.condition_time:
        #     # Slice off last dimension which represented time.
        #     h_final = h_final[:, :-1]

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
        # print(f"Shape of node_mask: {node_mask.shape}")
        beta = 1e-5
        x_recon_loss = torch.sum(node_mask.view(bs, n_nodes, 1) * (vel - x.view(bs, n_nodes, -1))**2) / torch.sum(node_mask)
    
        # Compute feature reconstruction loss if features exist
        h_recon_loss = torch.sum(node_mask.view(bs, n_nodes, 1) * (h_final.view(bs, n_nodes, -1) - h.view(bs, n_nodes, -1))**2) / torch.sum(node_mask)
        # Compute KL divergence loss using the DiagonalGaussianDistribution.kl() method
        x_kl_loss = encoded_batch['x_posterior'].kl().mean()
        h_kl_loss = encoded_batch['h_posterior'].kl().mean()
        
        # Total losses
        recon_loss = x_recon_loss + h_recon_loss
        kl_loss = x_kl_loss + h_kl_loss
        total_loss = recon_loss + beta * kl_loss
        
        loss_dict =  {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'x_recon_loss': x_recon_loss,
            'h_recon_loss': h_recon_loss
        }
        print(f"loss_dict: {loss_dict}")
        # Original return value (coordinates and features)
        model_output = torch.cat([vel, h_final.view(bs, n_nodes, -1)], dim=2)

        # Return both the model output and loss dictionary
        return model_output, loss_dict


        # debug = False
        # if h_dims == 0:
        #     return vel
        # elif debug:
        #     h_final = h_final.view(bs, n_nodes, -1)
        #     # print(f"Shape of h_final in stage 3: {h_final.shape}")
        #     return torch.cat([vel, h_final], dim=2), pose
        # else:
        #     h_final = h_final.view(bs, n_nodes, -1)
        #     # print(f"Shape of h_final in stage 3: {h_final.shape}")
        #     return torch.cat([vel, h_final], dim=2)

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