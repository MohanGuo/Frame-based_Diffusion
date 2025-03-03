import torch
import torch.nn as nn
import math
import numpy as np
from .transformer import Transformer
import sys
sys.path.append('..')
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask, assert_mean_zero_with_mask
from egnn.models import EGNN_dynamics_QM9_MC
from egnn.egnn_mc import EGNN as EGNN_mc
    
class TransformerDynamics_2(nn.Module):
    def __init__(self, args,in_node_nf, context_node_nf, n_dims, 
                 hidden_nf=64, device='cpu', n_heads=4, 
                 n_layers=4, condition_time=True):
        # (self, in_node_nf, context_node_nf,
        #          n_dims, hidden_nf=64, device='cpu',
        #          act_fn=torch.nn.SiLU(), n_layers=4, attention=False,
        #          condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
        #          inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'):
        
        super().__init__()
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        # self._edges_dict = {}
        self.condition_time = condition_time
        

        self.transformer = Transformer(
            args=args,
            in_node_nf=in_node_nf + context_node_nf, # + 1 if condition_time else in_node_nf + context_node_nf,
            device=device,
            d_model=hidden_nf,
            num_heads=n_heads,
            num_layers=n_layers,
            d_ff=hidden_nf*4,
            dropout=0.1
        )


        self.egnn = EGNN_dynamics_QM9_MC(in_node_nf=in_node_nf - 1, context_node_nf=args.context_node_nf,
                 n_dims=3, device=device, hidden_nf=args.nf,
                 act_fn=torch.nn.SiLU(), n_layers=3, attention=False,
                #  condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                #  inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'，
                num_vectors=7, num_vectors_out=2
                 )

    def forward(self, t, xh, node_mask, edge_mask, context=None):
        raise NotImplementedError

    def wrap_forward(self, node_mask, edge_mask, context):
        def fwd(time, state):
            return self._forward(time, state, node_mask, edge_mask, context)
        return fwd

    def unwrap_forward(self):
        return self._forward

    def _forward(self, t, xh, node_mask, edge_mask, context):
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        
        ################## MG: pass egnn to get invariant coordinates ###################
        # xh = torch.cat([x, h['categorical'], h['integer']], dim=2)
        x_input =remove_mean_with_mask(xh[..., :self.n_dims], node_mask)
        xh_input = torch.cat([x_input, xh[..., self.n_dims:]], dim=2)

        # with torch.no_grad():
        output, _, egnn_f = self.egnn._forward(xh_input, node_mask, edge_mask, context=None)
        # print(f"output in TransformerDynamics_2: {output[..., :self.n_dims]}")
        x_invariant = output[..., :self.n_dims]
        # print(f"Shape of x_invariant: {x_invariant.shape}")
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        x_invariant = x_invariant.view(bs*n_nodes, -1)
        #################################################################################

        # Add time after egnn
        # bs, n_nodes, dims = xh.shape
        
        xh = xh.view(bs*n_nodes, -1)#.clone() * node_mask
        x = xh[:, 0:self.n_dims].clone()
        if h_dims == 0:
            h = torch.ones(bs*n_nodes, 1).to(self.device)
        else:
            h = xh[:, self.n_dims:].clone()
        
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

        
        h_final, x_final = self.transformer(h, x_invariant, node_mask=node_mask, batch_size=bs)
        # h_final, x_final = h, x_invariant
        # vel = (x_final) * node_mask
        # print(f"x_invariant: {x_invariant}")

        # vel = (x_final - x_invariant) * node_mask
        
        ################# if transform back #####################
        # # print(f"Shape of x_final: {x_final.shape}")
        #x_final: [bs*nodes, 3]
        x_final_equivariant = torch.bmm(x_final.unsqueeze(1), egnn_f).squeeze(1)
        # x_final_equivariant = torch.bmm(egnn_f.permute(0,2,1), x_final.unsqueeze(-1)).squeeze(-1)
        # x_final_equivariant = torch.bmm(egnn_f, x_final.unsqueeze(2)).squeeze(2)
        vel = (x_final_equivariant ) * node_mask
        # # print(f"h_final: {h_final}")
        # print(f"x_final_equivariant: {x_final_equivariant}")
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
            print(f"Remove mean with mask!")

        # print(f"Shape of node_mask: {node_mask}")
        assert_mean_zero_with_mask(vel, node_mask.view(bs, n_nodes, 1))

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

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

@dataclass
class Args:
    attention_type: str = "performer"
    max_positions: int = 32
    num_landmarks: int = 32
    batch_size: int = 32

# Create a test function for the TransformerDynamics model
def test_transformer_dynamics():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    in_node_nf = 3  # Input node feature dimension
    context_node_nf = 0  # Context node feature dimension
    n_dims = 3  # Spatial dimensions
    hidden_nf = 64  # Hidden layer dimension
    n_heads = 4  # Number of attention heads
    n_layers = 4  # Number of layers
    batch_size = 32  # Batch size
    n_nodes = 5  # Number of nodes
    
    # Initialize model
    args = Args()
    model = TransformerDynamics(
        args=args,
        in_node_nf=in_node_nf,
        context_node_nf=context_node_nf,
        n_dims=n_dims,
        hidden_nf=hidden_nf,
        device=device,
        n_heads=n_heads,
        n_layers=n_layers
    ).to(device)
    
    # Create test inputs
    t = torch.tensor([0.5]).to(device)  # Time step
    
    # Create input features xh (containing position and node features)
    xh = torch.randn(batch_size, n_nodes, n_dims + in_node_nf).to(device)
    
    # Create masks
    node_mask = torch.ones(batch_size, n_nodes, 1).to(device)
    edge_mask = torch.ones(batch_size, n_nodes, n_nodes, 1).to(device)
    
    # Create context features
    context = None
    
    # Forward pass
    try:
        output = model._forward(t, xh, node_mask, edge_mask, context)
        print(f"Output shape: {output.shape}")
        print("Test passed successfully!")
        
        # Check for NaN values in output
        if torch.isnan(output).any():
            print("Warning: Output contains NaN values!")
        else:
            print("Output contains no NaN values.")
            
        # Verify output dimensions
        expected_output_dims = n_dims + in_node_nf  # -1 because of h_final = h_final[..., :-1]
        if output.shape[-1] == expected_output_dims:
            print(f"Output dimensions are correct: {output.shape}")
        else:
            print(f"Warning: Unexpected output dimensions. Expected {expected_output_dims}, got {output.shape[-1]}")
            
        return output
        
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        raise

if __name__ == "__main__":
    output = test_transformer_dynamics()