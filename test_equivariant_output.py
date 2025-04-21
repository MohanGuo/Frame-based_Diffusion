import torch
# from egnn.models_new import EGNN_dynamics_QM9
from model.transformer_dynamic import TransformerDynamics_2
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import sys
sys.path.append('..')
from qm9 import dataset
from types import SimpleNamespace
from configs.datasets_config import get_dataset_info
# torch.set_default_dtype(torch.float64)
# dtype = torch.float64
torch.random.manual_seed(17)
dtype = torch.float
def random_rotation_matrix():
    """Generate random 3D rotation matrix"""
    theta = torch.randn(3)
    theta = theta / torch.norm(theta)
    K = torch.tensor([[0, -theta[2], theta[1]],
                     [theta[2], 0, -theta[0]],
                     [-theta[1], theta[0], 0]])
    R = torch.matrix_exp(K)
    return R

def test_equivariance_2():
    # Model setup
    device = 'cpu'
    # dtype = torch.float
    cfg = SimpleNamespace(
        dataset='qm9',
        batch_size=32,
        num_workers=4,
        filter_n_atoms=None,
        datadir='qm9/temp',
        remove_h=False,
        include_charges=False,
        device=device,
        sequential=False,
        context_node_nf=0,
        nf=128
    )
    dataloaders, _ = dataset.retrieve_dataloaders(cfg)
    loader = dataloaders['train']
    dataset_info = get_dataset_info('qm9', False)

    dynamics_in_node_nf = len(dataset_info['atom_decoder']) + int(False) + 1
    context_node_nf = 0
    nf = 128
    n_layers = 6

    egnn = TransformerDynamics_2(
        args=cfg,
        in_node_nf=dynamics_in_node_nf, context_node_nf=context_node_nf,
        n_dims=3, device=device, hidden_nf=nf,
        n_heads=8,
        n_layers=n_layers,
        condition_time=True
        )
    # egnn = EGNN_dynamics_QM9_MC(
    #     in_node_nf=dynamics_in_node_nf, context_node_nf=context_node_nf,
    #     n_dims=3, device=device, hidden_nf=nf,
    #     act_fn=torch.nn.SiLU(), n_layers=n_layers,
    #     attention=True, 
    #     num_vectors=7, num_vectors_out=3
    #     )

    for data in loader:
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        print(f"Shape of node mask: {node_mask.shape}")
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = data['charges'].to(device, dtype)

        x = remove_mean_with_mask(x, node_mask)
        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)
        
        # Combine features
        xh = torch.cat([x, one_hot, charges], dim=2)
        print(f"Shape of xh: {xh.shape}")
        # Shape of xh: torch.Size([32, 27, 8])
        
        t = torch.tensor([0.5])
        # Original output
        out1 = egnn._forward(t, xh, node_mask, edge_mask, context=None)
        
        # Test rotation equivariance
        R = random_rotation_matrix()
        xh_rot = xh.clone()
        xh_rot[..., :3] = torch.matmul(xh[..., :3], R.T)  # Only rotate positions
        print(f"Shape of rotation matrix: {R.shape}")

        out2 = egnn._forward(t, xh_rot, node_mask, edge_mask, context=None)
        out2_unrot = out2.clone()
        
        
        ############# Test input invariance ##################

        print(f"Position of ")

        rot_error = torch.mean((out1[..., :3] - out2[..., :3]) ** 2)
        print(f"Rotation invariance error: {rot_error.item():.8f}")
        rot_error = torch.mean((out1[..., 3:-1] - out2[..., 3:-1]) ** 2)
        print(f"Rotation invariance error on h: {rot_error.item():.8f}")

        out1_rot = torch.matmul(out1[..., :3], R.T)
        # print(f"Shape of out1_rot: {out1_rot.shape}")
        # output_equivariance = torch.allclose(out1_rot, out2[..., :3], rtol=1e-5, atol=1e-5)
        # print(f"Rotation Output equivariance: {output_equivariance}")
        rot_error = torch.mean((out1_rot - out2[..., :3]) ** 2)
        print(f"Rotation equivariance error: {rot_error.item():.8f}")
        

        break

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

if __name__ == "__main__":
    test_equivariance_2()
    # test_equivariance()