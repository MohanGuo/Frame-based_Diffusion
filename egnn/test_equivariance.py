import torch
from models import EGNN_dynamics_QM9
import sys
sys.path.append('..')
from qm9 import dataset
from types import SimpleNamespace
from configs.datasets_config import get_dataset_info

def random_rotation_matrix():
    """Generate random 3D rotation matrix"""
    theta = torch.randn(3)
    theta = theta / torch.norm(theta)
    K = torch.tensor([[0, -theta[2], theta[1]],
                     [theta[2], 0, -theta[0]],
                     [-theta[1], theta[0], 0]])
    R = torch.matrix_exp(K)
    return R

def test_equivariance():
    # Model setup
    device = 'cpu'
    dtype = torch.float
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
        
        # Combine features
        xh = torch.cat([x, one_hot, charges], dim=2)
        print(f"Shape of xh: {xh.shape}")
        
        # Original output
        out1 = egnn._forward(xh, node_mask, edge_mask, context=None)
        
        # Test rotation equivariance
        R = random_rotation_matrix()
        xh_rot = xh.clone()
        xh_rot[..., :3] = torch.matmul(xh[..., :3], R.T)  # Only rotate positions
        print(f"Shape of rotation matrix: {R.shape}")

        out2 = egnn._forward(xh_rot, node_mask, edge_mask, context=None)
        out2_unrot = out2.clone()
        out2_unrot[..., :3] = torch.matmul(out2[..., :3], R)
        
        rot_error = torch.mean((out1[..., :3] - out2_unrot[..., :3]) ** 2)
        print(f"Rotation equivariance error: {rot_error.item():.8f}")

        out1_rot = torch.matmul(out1[..., :3], R.T)
        print(f"Shape of out1_rot: {out1_rot.shape}")
        output_equivariance = torch.allclose(out1_rot, out2[..., :3], rtol=1e-5, atol=1e-5)
        print(f"Output equivariance: {output_equivariance}")
        
        # Test translation equivariance
        t = torch.randn(3)
        xh_trans = xh.clone()
        xh_trans[..., :3] += t  # Only translate positions
        out3 = egnn._forward(xh_trans, node_mask, edge_mask, context=None)
        
        trans_error = torch.mean((out1 - out3) ** 2)
        print(f"Translation equivariance error: {trans_error.item():.8f}")
        break

if __name__ == "__main__":
    test_equivariance()