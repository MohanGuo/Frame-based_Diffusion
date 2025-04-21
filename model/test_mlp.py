import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import torch.nn as nn
import numpy as np

from data.dataset import QM9Wrapper
from egnn.egnn import EGNN



def torch_cov(x):
    x_centered = x - x.mean(dim=0)
    factor = 1 / (x.size(0) - 1)
    return factor * (x_centered.T @ x_centered)


class SimpleMLP(nn.Module):
    """Simple MLP for processing transformed coordinates and node features"""
    def __init__(self, node_dim, hidden_dim=128):
        """
        Args:
            node_dim (int): Dimension of node features
            hidden_dim (int): Hidden layer dimension
        """
        super(SimpleMLP, self).__init__()
        
        coord_dim = 3  # 3D coordinates
        input_dim = coord_dim + node_dim  # Combined input dimension
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, coord_dim)  # Output same dimension as coordinates
        )
        
    def forward(self, x_transformed, h):
        """
        Args:
            x_transformed (torch.Tensor): Transformed coordinates [N, 3]
            h (torch.Tensor): Node features [N, node_dim]
            
        Returns:
            torch.Tensor: Output coordinates [N, 3]
        """
        # Concatenate transformed coordinates and node features
        combined = torch.cat([x_transformed, h], dim=-1)
        
        # Process through MLP
        out = self.mlp(combined)
        
        return out

def test_equivariance(egnn, h, x, edge_index, edge_feat=None, model=None):
    """
    Test E(3) equivariance of the model

    Args:
        model: EGNN model
        h: Node features [n_nodes, input_nf]
        x: Node coordinates [n_nodes, 3] 
        edges: Edge indices [2, n_edges]
        
    Test both rotation equivariance and translation invariance
    """
    # Original forward pass
    # print(f"Original coordinates: {x}")
    # print(f"Original features: {h}")

    h1, f1 = egnn(h, x, edge_index, edge_feat) # f1: [19, 3]

    ## Generate random rotation and rotate input
    R = random_rotation_matrix()
    R = torch.tensor(R, dtype=torch.float32)
    x_rotated = torch.matmul(x, R.T)
    # print(f"Rotated coordinates: {x_rotated}")
    h2, f2 = egnn(h, x_rotated, edge_index, edge_feat)

    transformed_f1 = f1[:3, :]
    # print(f"Shape of transformed_f1: {transformed_f1.shape}")
    # print(f"Transformed f1: {transformed_f1}")
    transformed_f1_inverse = torch.inverse(transformed_f1)
    input_x1 = torch.matmul(x, transformed_f1_inverse)
    print(f"Input x1: {input_x1}")
    output1 = model(input_x1, h)
    result1 = torch.matmul(output1, transformed_f1)
    print(f"Original result: {result1}")


    ## Test rotation equivariance
    transformed_f2 = f2[:3, :]
    # print(f"Shape of transformed_f2: {transformed_f2.shape}")
    # print(f"Transformed f2: {transformed_f2}")
    transformed_f1_rotated = torch.matmul(transformed_f1, R.T)
    # print(f"Transformed f1 rotated: {transformed_f1_rotated}")
    # print(f"f2: {f2}")
    transformed_f2_inverse = torch.inverse(transformed_f2)
    #    print(f"Shape ofleft_inv: {left_inv.shape}")
    #    print(f"Shape of x: {x.shape}")
    # print(f"Transformed f2 inverse: {transformed_f2_inverse}")
    # print(f"Rotated transformed f1 inverse: {torch.matmul(torch.inverse(R.T), transformed_f1_inverse)}")
    input_x2 = torch.matmul(x_rotated, transformed_f2_inverse)
    print(f"Input x2: {input_x2}")
    print(f"Imagined output x1: {torch.matmul(torch.matmul(x, R.T), torch.matmul(torch.inverse(R.T), transformed_f1_inverse))}")
    #    input_x = left_inv * x
    # print(f"Transfromed f1: {transformed_f1}")
    #    print(f"Shape of input_x: {input_x.shape}")
    #    print(f"Shape of h: {h.shape}")
    output2 = model(input_x2, h)
    #    print(f"Original model output: {output1}, shape: {output1.shape}")
    #    f = torch.broadcast_to(row_sums, (19, 19))
    #    result1 = torch.matmul(f, output1)
    result2 = torch.matmul(output2, transformed_f2)
    result1_rotated = torch.matmul(result1, R.T)
    print(f"Rotated result: {result2}")
    print(f"Rotated original result: {result1_rotated}")
    #    print(f"Original output features: {h1}")
    #    print(f"Original output coordinates: {x1}")

    ## Input should be invariant
    is_equal = torch.allclose(input_x1, input_x2, rtol=1e-5, atol=1e-4)
    print(f"Input x1 and x2 are equal: {is_equal}")

    ## Output should be equivariant, so the rotated output should be equal
    is_equal = torch.allclose(result1_rotated, result2, rtol=1e-5, atol=1e-5)
    print(f"Results are equal: {is_equal}")


    # Rotate original output
    # output1_rotated = torch.matmul(output1, R.T)
    # print(f"Rotated original coordinates: {output1_rotated}")

    #    print("Equivariance test results:")
    #    print(f"Max node feature difference: {torch.max(torch.abs(h1 - h2)).item():.6f}")
    #    print(f"Max coordinate difference: {torch.max(torch.abs(x1_rotated - x2)).item():.6f}")

def random_rotation_matrix():
    """
    Generate a random 3D rotation matrix

    Returns:
        R: 3x3 rotation matrix as numpy array
    """
    theta = np.random.rand() * 2 * np.pi
    phi = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2

    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    return R

def analyze_atom_distribution(qm9_wrapper):
    """
    Analyze the distribution of atom counts in the QM9 dataset.
    
    Args:
        qm9_wrapper (QM9Wrapper): Instance of QM9Wrapper class
        
    Returns:
        dict: Dictionary containing:
            - min_atoms: Minimum number of atoms in any molecule
            - max_atoms: Maximum number of atoms in any molecule
            - mean_atoms: Average number of atoms across all molecules
            - atom_histogram: Dictionary mapping atom counts to frequency
    """
    # Get the dataset
    dataset = qm9_wrapper.dataset
    
    # Initialize counters
    atom_counts = []
    atom_histogram = {}
    
    # Collect statistics
    for data in dataset:
        num_atoms = data.num_nodes
        atom_counts.append(num_atoms)
        atom_histogram[num_atoms] = atom_histogram.get(num_atoms, 0) + 1
    
    # Calculate statistics
    stats = {
        'min_atoms': min(atom_counts),
        'max_atoms': max(atom_counts),
        'mean_atoms': sum(atom_counts) / len(atom_counts),
        'atom_histogram': dict(sorted(atom_histogram.items()))
    }
    
    # Print summary
    print("QM9 Dataset Atom Distribution:")
    print(f"Minimum atoms per molecule: {stats['min_atoms']}")
    print(f"Maximum atoms per molecule: {stats['max_atoms']}")
    print(f"Average atoms per molecule: {stats['mean_atoms']:.2f}")
    print("\nHistogram of atom counts:")
    for num_atoms, count in stats['atom_histogram'].items():
        print(f"{num_atoms} atoms: {count} molecules")
        
    return stats


if __name__ == "__main__":

    qm9 = QM9Wrapper()
    
    analyze_atom_distribution(qm9)

    # Get test batch
    batch = qm9.get_test_batch(num_samples=5)

    # Model parameters
    model_params = {
        'in_size': batch.x.size(1),
        'hidden_size': 128,
        'out_size': 64,
        'n_layers': 10
        # 'edge_feat_size': batch.edge_attr.size(1)
    }


    # Initialize model
    egnn = EGNN(**model_params)

    print("Extracting batch data...")
    h = batch.x.float()
    x = batch.pos
    edges = batch.edge_index
    edge_feat = batch.edge_attr
    print(f"Number of nodes: {h.size(0)}")
    print(f"Number of edges: {edges.size(1)}")




    # Test dimensions
    batch_size = 5
    node_dim = 11

    # Create model
    model = SimpleMLP(node_dim=node_dim, hidden_dim=128)

    print("\nStarting equivariance test...")
    # Test rotation equivariance
    test_equivariance(egnn, h, x, edges, edge_feat, model)