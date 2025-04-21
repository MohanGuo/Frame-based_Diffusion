import torch
from egnn.models import EGNN_dynamics_QM9, EGNN_dynamics_QM9_MC
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

def test_equivariance():
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
        sequential=False
    )
    dataloaders, _ = dataset.retrieve_dataloaders(cfg)
    loader = dataloaders['train']
    dataset_info = get_dataset_info('qm9', False)

    dynamics_in_node_nf = len(dataset_info['atom_decoder']) + int(False)
    context_node_nf = 0
    nf = 128
    n_layers = 6

    # egnn = EGNN_dynamics_QM9(
    #     in_node_nf=dynamics_in_node_nf, context_node_nf=context_node_nf,
    #     n_dims=3, device=device, hidden_nf=nf,
    #     act_fn=torch.nn.SiLU(), n_layers=n_layers,
    #     attention=True, tanh=True, mode=None, norm_constant=1,
    #     inv_sublayers=1, sin_embedding=False,
    #     normalization_factor=1, aggregation_method='sum')
    egnn = EGNN_dynamics_QM9_MC(
        in_node_nf=dynamics_in_node_nf, context_node_nf=context_node_nf,
        n_dims=3, device=device, hidden_nf=nf,
        act_fn=torch.nn.SiLU(), n_layers=n_layers,
        attention=True, 
        num_vectors=7, num_vectors_out=3
        )

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
        
        # Original output
        out1, vel_inverse1, x_final1 = egnn._forward(xh, node_mask, edge_mask, context=None)
        
        # Test rotation equivariance
        R = random_rotation_matrix()
        xh_rot = xh.clone()
        xh_rot[..., :3] = torch.matmul(xh[..., :3], R.T)  # Only rotate positions
        print(f"Shape of rotation matrix: {R.shape}")

        out2, vel_inverse2, x_final2 = egnn._forward(xh_rot, node_mask, edge_mask, context=None)
        out2_unrot = out2.clone()
        # vel_inverse2 = vel_inverse2.permute(0, 2, 1)
        # vel_inverse1 = vel_inverse1.permute(0, 2, 1)
        vel_inverse1_rot = torch.einsum('ij,njs->nis', (torch.inverse(R.T), vel_inverse1))
        # vel_inverse1_rot = torch.einsum('ij,njs->nis', R, vel_inverse1)
        # vel_inverse1_rot = torch.matmul(torch.inverse(R.T), vel_inverse1)  # Only rotate positions('ij,njs->nis', (torch.inverse(R.T), vel_inverse1))
        print(f"Shape of vel_inverse1_rot: {vel_inverse1_rot.shape}")
        print(f"Rotation inverse: {torch.mean((vel_inverse1_rot[:, :, 0] - vel_inverse2[:, :, 0]) ** 2)}")

        vel_inverse1_rot = torch.matmul(torch.inverse(R.T), vel_inverse1[0,...])
        print(f"Rotation inverse: {torch.mean((vel_inverse1_rot - vel_inverse2[0,...]) ** 2)}")

        ########### Test inverse ######################
        
        print(f"Shape of x_final1: {x_final1.shape}")
        x_final1_rot = torch.einsum('njs,ji->nis', x_final1, R.T)
        print(f"Rotation of final positions in egnn output: {torch.mean((x_final1_rot - x_final2) ** 2)}")

        x_final1_0_rot = torch.matmul((x_final1[0,...]).T, R.T)
        x_final2_0 = (x_final2[0,...].T)
        print(f"Rotation of final positions in one node of egnn output: {torch.mean((x_final1_0_rot - x_final2_0) ** 2)}")

        Q1 = torch.linalg.qr(x_final1[0,...])[0].T
        print(f"Q1: {Q1}")
        for i in range(Q1.shape[0]):
            if torch.dot(Q1[i], x_final1[0,...].T[i]) < 0:
                Q1[i] = -Q1[i]
        x_final1_0_gs = Q1

        Q2 = torch.linalg.qr(x_final2[0,...])[0].T
        print(f"Q2: {Q2}")
        for i in range(Q2.shape[0]):
            if torch.dot(Q2[i], x_final2[0,...].T[i]) < 0:
                Q2[i] = -Q2[i]
        x_final2_0_gs = Q2

        x_final1_0_gs_rot = torch.matmul(x_final1_0_gs, R.T)
        print(f"x_final1[0,...]: {x_final1[0,...]}")
        print(f"x_final2[0,...]: {x_final2[0,...]}")
        print(f"x_final1_0_gs: {x_final1_0_gs}")
        print(f"x_final1_0_gs_rot: {x_final1_0_gs_rot}")
        print(f"x_final2_0_gs: {x_final2_0_gs}")
        print(f"Rotation of final positions in one node of egnn output after GS: {torch.mean((x_final1_0_gs_rot + x_final2_0_gs) ** 2)}")
        
        x_final1_0_gs_rot_det = torch.det(x_final1_0_gs_rot)
        x_final2_0_gs_det = torch.det(x_final2_0_gs)
        print(f"x_final1_0_gs_rot_det: {x_final1_0_gs_rot_det}")
        print(f"x_final2_0_gs_det: {x_final2_0_gs_det}")
        # P1 = x_final1_0_gs_rot @ x_final1_0_gs_rot.T
        # P2 = x_final2_0_gs @ x_final2_0_gs.T
        # error = torch.norm(P1 - P2)
        error = torch.min(
            torch.norm(x_final1_0_gs_rot - x_final2_0_gs),
            torch.norm(x_final1_0_gs_rot + x_final2_0_gs)
        )
        print(f"Error: {error:.2e}")
        # det_x_final1 = torch.det(x_final1[0,...])
        # det_x_final2 = torch.det(x_final2[0,...])
        # print(f"Determinant of x_final1[0,...]: {det_x_final1}")
        # print(f"Determinant of x_final2[0,...]: {det_x_final2}")
        print(f"Determinant of GS: {torch.det(torch.linalg.qr(x_final1[0,...].T)[0])}")
        x_final1_0_inverse = torch.inverse(x_final1_0_gs)
        x_final1_0_inverse_rot = torch.matmul(torch.inverse(R.T), x_final1_0_inverse)
        x_final2_0_inverse = torch.inverse(x_final2_0_gs)
        print(f"Rotation of final positions of inversed one node: {torch.mean((x_final1_0_inverse_rot - x_final2_0_inverse) ** 2)}")

        
        ############# Test input invariance ##################

        rot_error = torch.mean((out1[..., :3] - out2[..., :3]) ** 2)
        print(f"Rotation invariance error: {rot_error.item():.8f}")
        rot_error = torch.mean((out1[..., 3:-1] - out2[..., 3:-1]) ** 2)
        print(f"Rotation invariance error on h: {rot_error.item():.8f}")

        # out1_rot = torch.matmul(out1[..., :3], R.T)
        # print(f"Shape of out1_rot: {out1_rot.shape}")
        # output_equivariance = torch.allclose(out1_rot, out2[..., :3], rtol=1e-5, atol=1e-5)
        # print(f"Output equivariance: {output_equivariance}")
        
        ################ Test translation equivariance #################
        t = torch.randn(3) * 100
        t = t * node_mask
        xh_trans = xh.clone()
        print(f"t: {t.shape}")
        xh_trans[..., :3] += t  # Only translate positions

        xh_trans[..., :3] = remove_mean_with_mask(xh_trans[..., :3], node_mask)
        check_mask_correct([xh_trans[..., :3], one_hot, charges], node_mask)

        out3, vel_inverse3, x_final3 = egnn._forward(xh_trans, node_mask, edge_mask, context=None)
        print(f"Shape of x_final3: {x_final3.shape}")
        # x_final3 = x_final3.permute(0, 2, 1)
        # x_final1 = x_final1.permute(0, 2, 1)
        trans_error = torch.mean((out1 - out3) ** 2)
        # trans_error = torch.mean((x_final1 - (x_final3 - t)) ** 2)
        # trans_error = torch.mean((x_final1[..., 0] - (x_final3[..., 0] - t)) ** 2)
        print(f"Translation equivariance error: {trans_error.item():.8f}")
        break


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

if __name__ == "__main__":
    test_equivariance_2()
    # test_equivariance()