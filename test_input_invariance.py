import torch
from models import EGNN_dynamics_QM9
import sys
sys.path.append('..')
from qm9 import dataset
from types import SimpleNamespace
from configs.datasets_config import get_dataset_info
import numpy as np

def random_rotation_matrix():
    theta = np.random.rand() * 2 * np.pi
    phi = np.random.rand() * 2 * np.pi
    
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return R

class SimpleMLP(torch.nn.Module):
    def __init__(self, node_dim, hidden_dim=128):
        super(SimpleMLP, self).__init__()
        coord_dim = 3
        input_dim = coord_dim + node_dim
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, coord_dim)
        )
        
    def forward(self, x_transformed, h):
        combined = torch.cat([x_transformed, h], dim=-1)
        return self.mlp(combined)
torch.set_default_dtype(torch.float64)
def test_equivariance():
    device = 'cpu'
    dtype = torch.float64
    cfg = SimpleNamespace(
        dataset='qm9',
        batch_size=32,
        num_workers=4,
        filter_n_atoms=None,
        datadir='qm9/temp',
        remove_h=False,
        include_charges=False,
        device=device,
        sequential=False
    )
    
    dataloaders, _ = dataset.retrieve_dataloaders(cfg)
    loader = dataloaders['train']
    dataset_info = get_dataset_info('qm9', False)
    
    dynamics_in_node_nf = len(dataset_info['atom_decoder']) + int(False)
    context_node_nf = 0
    nf = 128
    n_layers = 6
    
    egnn = EGNN_dynamics_QM9(
        in_node_nf=dynamics_in_node_nf, context_node_nf=context_node_nf,
        n_dims=3, device=device, hidden_nf=nf,
        act_fn=torch.nn.SiLU(), n_layers=n_layers,
        attention=True, tanh=True, mode=None, norm_constant=1,
        inv_sublayers=1, sin_embedding=False,
        normalization_factor=1, aggregation_method='sum')
    
    for data in loader:
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)
        
        # Combine features for EGNN input
        h = torch.cat([one_hot], dim=-1)
        
        # Create SimpleMLP model
        node_dim = h.size(-1)
        model = SimpleMLP(node_dim=node_dim, hidden_dim=128)
        
        # Original forward pass
        f1 = egnn._forward(torch.cat([x, h], dim=-1), node_mask, edge_mask, context=None)
        print(f"Shape of original output: {f1.shape}")
        
        # Generate random rotation and rotate input
        R = torch.tensor(random_rotation_matrix(), dtype=dtype)
        print(f"Rotation matrix: {R.shape}")
        x_rotated = torch.matmul(x, R.T)
        f2 = egnn._forward(torch.cat([x_rotated, h], dim=-1), node_mask, edge_mask, context=None)

        # print(f"Rotated output: {torch.matmul(f1[..., :3], R.T) - f2[..., :3]}")
        rotated_f1 = torch.matmul(f1[..., :3], R.T)
        # print(f"Rotated EGNN output: {}")
        output_equivariance = torch.allclose(rotated_f1, f2[..., :3], rtol=1e-5, atol=1e-5)
        print(f"Output EGNN equivariance: {output_equivariance}")
        
        # Extract transformation matrices
        transformed_f1 = f1[:, :3, :3]
        transformed_f2 = f2[:, :3, :3]

        print(f"shape of transformed_f1: {transformed_f1.shape}")
        
        # Test equivariance
        transformed_f1_inverse = torch.inverse(transformed_f1)
        transformed_f2_inverse = torch.inverse(transformed_f2)
        
        # Calculate inputs for SimpleMLP
        input_x1 = torch.matmul(x, transformed_f1_inverse)
        input_x2 = torch.matmul(x_rotated, transformed_f2_inverse)
        
        # Get outputs and apply transformations
        output1 = model(input_x1, h)
        output2 = model(input_x2, h)
        
        result1 = torch.matmul(output1, transformed_f1)
        result2 = torch.matmul(output2, transformed_f2)
        
        # Rotate original result for comparison
        result1_rotated = torch.matmul(result1, R.T)
        
        # Check invariance of inputs
        # print(f"Difference of inputs: {input_x1 - input_x2}")
        # print(f"Input x1: {input_x1}")
        input_invariance = torch.allclose(input_x1, input_x2, rtol=1e-4, atol=1e-5)
        print(f"Input invariance: {input_invariance}")
        
        # Check equivariance of outputs
        output_equivariance = torch.allclose(result1_rotated, result2, rtol=1e-4, atol=1e-5)
        print(f"Output equivariance: {output_equivariance}")
        break

def test_singular_case():
    # singular f
    f = torch.tensor([
        [1.0, 0.0, 1.0], 
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0]
    ], dtype=torch.float64)
    
    x = torch.randn(10, 3, dtype=torch.float64)
    h = torch.randn(10, 5, dtype=torch.float64)
    
    model = SimpleMLP(node_dim=5, hidden_dim=128)
    
    R = torch.tensor(random_rotation_matrix(), dtype=torch.float64)
    
    x_rotated = torch.matmul(x, R.T)
    f_rotated = torch.matmul(f, R.T)
    
    f_pinv = torch.pinverse(f)
    f_rotated_pinv = torch.pinverse(f_rotated)
    
    # print(f"Rank of f: {torch.matrix_rank(f)}")
    # print(f"Condition number of f: {torch.linalg.cond(f)}")
    
    input_x1 = torch.matmul(x, f_pinv)
    input_x2 = torch.matmul(x_rotated, f_rotated_pinv)
    
    output1 = model(input_x1, h)
    output2 = model(input_x2, h)
    
    result1 = torch.matmul(output1, f)
    result2 = torch.matmul(output2, f_rotated)
    
    result1_rotated = torch.matmul(result1, R.T)
    
    input_invariance = torch.allclose(input_x1, input_x2, rtol=1e-4, atol=1e-5)
    output_equivariance = torch.allclose(result1_rotated, result2, rtol=1e-4, atol=1e-5)
    
    print("\nInput difference:")
    print((torch.abs(input_x1 - input_x2)))
    # print(torch.mean(torch.abs(input_x1 - input_x2)))

    print(f"Input invariance: {input_invariance}")
    print(f"Output equivariance: {output_equivariance}")

if __name__ == "__main__":
    # test_equivariance()
    test_singular_case()