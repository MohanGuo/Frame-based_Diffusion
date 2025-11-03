import torch
# from egnn.models_new import EGNN_dynamics_QM9
from egnn.models import EGNN_dynamics_QM9_MC
# from model.transformer_dynamic_dit import TransformerDynamics_2
# from model.transformer_dynamic_conditional import TransformerDynamics_2
from model.transformer_dynamic_conditional import TransformerDynamics_2
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import sys
sys.path.append('..')
from qm9 import dataset
from types import SimpleNamespace
from configs.datasets_config import get_dataset_info
from test_rotation import visualize_molecule
# torch.set_default_dtype(torch.float64)
# dtype = torch.float64
torch.random.manual_seed(1)
dtype = torch.float
# def random_rotation_matrix():
#     """Generate random 3D rotation matrix"""
#     theta = torch.randn(3)
#     theta = theta / torch.norm(theta)
#     K = torch.tensor([[0, -theta[2], theta[1]],
#                      [theta[2], 0, -theta[0]],
#                      [-theta[1], theta[0], 0]])
#     R = torch.matrix_exp(K)
#     return R
def random_rotation_matrices(batch_size, device):
    """Generate random 3D rotation matrices for each sample in the batch"""
    rotation_matrices = []
    
    for _ in range(batch_size):
        # 
        theta = torch.randn(3, device=device)
        theta = theta / torch.norm(theta)  # 
        
        #  K
        K = torch.zeros((3, 3), device=device)
        K[0, 1], K[0, 2] = -theta[2], theta[1]
        K[1, 0], K[1, 2] = theta[2], -theta[0]
        K[2, 0], K[2, 1] = -theta[1], theta[0]
        
        #  R = exp(K)
        R = torch.matrix_exp(K)
        rotation_matrices.append(R)
    
    return torch.stack(rotation_matrices)  #  [batch_size, 3, 3] 

def test_equivariance_with_seed(seed):
    """"""
    # 
    torch.random.manual_seed(seed)

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
    n_layers = 8

    debug = False
    egnn = EGNN_dynamics_QM9_MC(in_node_nf=dynamics_in_node_nf - 1, context_node_nf=context_node_nf,
                 n_dims=3, device=device, hidden_nf=nf,
                 act_fn=torch.nn.SiLU(), n_layers=3, attention=False,
                #  condition_time=True, tanh=False, mode='egnn_dynamics', norm_constant=0,
                #  inv_sublayers=2, sin_embedding=False, normalization_factor=100, aggregation_method='sum'，
                num_vectors=7, num_vectors_out=2
                 )
    

    model = TransformerDynamics_2(
        args=cfg,
        egnn=egnn,
        in_node_nf=dynamics_in_node_nf, context_node_nf=context_node_nf,
        n_dims=3, device=device, hidden_nf=nf,
        n_heads=8,
        n_layers=n_layers,
        condition_time=True
        )
    
    results = {
        'rotation_invariance_error': 0.0,
        'rotation_invariance_error_h': 0.0,
        'rotation_equivariance_error': 0.0
    }

    model.eval()
    with torch.no_grad():
        for data in loader:
            x = data['positions'].to(device, dtype)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            print(f"Shape of node mask: {node_mask.shape}")
            # Shape of node mask: torch.Size([32, 25, 1])
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
            
            # t = torch.tensor([0.5])
            t = torch.randint(0, 1, size=(x.size(0), 1), device=x.device).float()
            # t = torch.zeros((x.size(0), 1), device=x.device).float()
            # Original output
            visualize_molecule(xh[..., :3], node_mask, batch_index=0, save_path='molecule_1_input.png')
            out1, _ = model._forward(t, xh, node_mask, edge_mask, context=None)
            print(f"out1 shape: {out1.shape}")
            visualize_molecule(out1[..., :3], node_mask, batch_index=0, save_path='molecule_1.png')
            
            # Test rotation equivariance
            batch_size = x.size(0)  # 32
            rotation_matrices = random_rotation_matrices(batch_size, x.device)
            # print(f"Shape of rotation matrices: {rotation_matrices.shape}")  # [bs, 3, 3]
            # rotation_matrix = torch.tensor([
            #     [-1.0, 0.0, 0.0],
            #     [0.0, -1.0, 0.0],
            #     [0.0, 0.0, 1.0]
            # ], device=device)
            # # Repeat the matrix for each item in the batch
            # rotation_matrices = rotation_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
    
            xh_rot = xh.clone()
            # xh[..., :3]  [batch_size, num_nodes, 3]
            #  [batch_size, num_nodes, 3] @ [batch_size, 3, 3]
            xh_rot[..., :3] = torch.bmm(xh[..., :3], rotation_matrices.transpose(1, 2))

            visualize_molecule(xh_rot[..., :3], node_mask, batch_index=0, save_path='molecule_2_input.png')
            out2, _ = model._forward(t, xh_rot, node_mask, edge_mask, context=None)
            out2_unrot = out2.clone()
            visualize_molecule(out2[..., :3], node_mask, batch_index=0, save_path='molecule_2.png')
            
            
            ############# Test input invariance ##################
            weight = 1e6

            # print(f"Position of original output: {out1[..., :3]}")
            # print(f"Position of rotated output: {out2[..., :3]}")
            

            rot_error = torch.mean(((out1[..., :3] - out2[..., :3]) * weight) ** 2)
            results['rotation_invariance_error'] = rot_error.item()
            print(f"A Rotation invariance error: {rot_error.item():.8f}")

            rot_error = torch.mean(((out1[..., 3:-1] - out2[..., 3:-1]) * weight) ** 2)
            results['rotation_invariance_error_h'] = rot_error.item()
            print(f"A Rotation invariance error on h: {rot_error.item():.8f}")

            out1_rot = torch.matmul(out1[..., :3], rotation_matrices.transpose(1, 2))
            error_tensor = ((out1_rot - out2[..., :3]) * weight) ** 2
            rot_error = torch.mean(error_tensor)
            results['rotation_equivariance_error'] = rot_error.item()
            print(f"A Rotation equivariance error: {rot_error.item():.8f}")

            absolute_diff = out1[..., :3] - out2[..., :3]
            norm_out1 = torch.norm(out1[..., :3], dim=2, keepdim=True).clamp(min=1e-8)
            relative_error = torch.mean((torch.norm(absolute_diff, dim=2) / norm_out1.squeeze(-1) )**2)
            results['rotation_invariance_error'] = relative_error.item()
            
            # 2. 
            if out1.shape[2] > 3:  # 
                absolute_diff_h = out1[..., 3:-1] - out2[..., 3:-1]
                norm_out1_h = torch.norm(out1[..., 3:-1], dim=2, keepdim=True).clamp(min=1e-8)
                relative_error_h = torch.mean((torch.norm(absolute_diff_h, dim=2) / norm_out1_h.squeeze(-1) * 1)**2)
                results['rotation_invariance_error_h'] = relative_error_h.item()
            else:
                results['rotation_invariance_error_h'] = 0.0
            
            # 3. 
            # out1_rot = torch.bmm(out1[..., :3], rotation_matrices)
            absolute_diff_equiv = out1_rot - out2[..., :3]
            norm_out1_rot = torch.norm(out1_rot, dim=2, keepdim=True).clamp(min=1e-8)
            relative_error_equiv = torch.mean((torch.norm(absolute_diff_equiv, dim=2) / norm_out1_rot.squeeze(-1) * 1)**2)
            results['rotation_equivariance_error'] = relative_error_equiv.item()

            

            # print(f"Position of rotated original output: {out1_rot}")

            # # Find the maximum error
            # max_error_value, max_indices = torch.max(error_tensor.view(-1), dim=0)
            # # Convert flat index back to 3D indices
            # batch_size, num_nodes, dims = error_tensor.shape
            # max_batch_idx = max_indices.item() // (num_nodes * dims)
            # remainder = max_indices.item() % (num_nodes * dims)
            # max_node_idx = remainder // dims
            # max_dim_idx = remainder % dims
            
            # # Print detailed information about the maximum error
            # print("\n=== Maximum Rotation Equivariance Error Details ===")
            # print(f"Maximum error value: {max_error_value.item():.8f}")
            # print(f"Location: batch={max_batch_idx}, node={max_node_idx}, dimension={max_dim_idx}")
            
            # # Print the actual values at this location
            # value1 = out1_rot[max_batch_idx, max_node_idx, max_dim_idx].item()
            # value2 = out2[max_batch_idx, max_node_idx, max_dim_idx].item()
            # print(f"Rotated output value: {value1:.8f}")
            # print(f"Direct output value: {value2:.8f}")
            # print(f"Absolute difference: {abs(value1 - value2):.8f}")

            ################ Test translation equivariance #################
            # trans = torch.randn(3) * 100
            # trans = trans * node_mask
            # xh_trans = xh.clone()
            # print(f"t: {t.shape}")
            # xh_trans[..., :3] += trans  # Only translate positions

            # xh_trans[..., :3] = remove_mean_with_mask(xh_trans[..., :3], node_mask)
            # check_mask_correct([xh_trans[..., :3], one_hot, charges], node_mask)

            # out3 = model._forward(t, xh_trans, node_mask, edge_mask, context=None)
            # # print(f"Shape of x_final3: {x_final3.shape}")
            # # x_final3 = x_final3.permute(0, 2, 1)
            # # x_final1 = x_final1.permute(0, 2, 1)
            # trans_error = torch.mean((out1 - out3) ** 2)
            # # trans_error = torch.mean((x_final1 - (x_final3 - t)) ** 2)
            # # trans_error = torch.mean((x_final1[..., 0] - (x_final3[..., 0] - t)) ** 2)
            # print(f"Translation equivariance error: {trans_error.item():.8f}")
            
            # # Analyze output scale and properties
            # print("\n=== Output Scale Analysis ===")
            
            # # Spatial components (first 3 dimensions)
            # spatial_out = out1[..., :3]
            # print(f"Spatial output shape: {spatial_out.shape}")
            # print(f"Spatial output min: {spatial_out.min().item():.6f}")
            # print(f"Spatial output max: {spatial_out.max().item():.6f}")
            # print(f"Spatial output mean: {spatial_out.mean().item():.6f}")
            # print(f"Spatial output std: {spatial_out.std().item():.6f}")
            
            # # Compute L2 norm of spatial components (velocity magnitude)
            # spatial_norm = torch.norm(spatial_out, dim=2)
            # print(f"Spatial L2 norm min: {spatial_norm.min().item():.6f}")
            # print(f"Spatial L2 norm max: {spatial_norm.max().item():.6f}")
            # print(f"Spatial L2 norm mean: {spatial_norm.mean().item():.6f}")
            
            # # If there are non-spatial features, analyze them too
            # if out1.shape[2] > 3:
            #     non_spatial_out = out1[..., 3:]
            #     print(f"\nNon-spatial output shape: {non_spatial_out.shape}")
            #     print(f"Non-spatial output min: {non_spatial_out.min().item():.6f}")
            #     print(f"Non-spatial output max: {non_spatial_out.max().item():.6f}")
            #     print(f"Non-spatial output mean: {non_spatial_out.mean().item():.6f}")
            #     print(f"Non-spatial output std: {non_spatial_out.std().item():.6f}")
            
            # # Analyze distribution per dimension
            # print("\nDistribution across spatial dimensions:")
            # for i in range(3):
            #     dim_data = spatial_out[..., i]
            #     print(f"Dimension {i}: min={dim_data.min().item():.6f}, max={dim_data.max().item():.6f}, " 
            #           f"mean={dim_data.mean().item():.6f}, std={dim_data.std().item():.6f}")
            
            # # Compare input and output scales
            # input_scale = torch.norm(x, dim=2).mean().item()
            # output_scale = spatial_norm.mean().item()
            # print(f"\nInput position scale (mean L2 norm): {input_scale:.6f}")
            # print(f"Output scale (mean L2 norm): {output_scale:.6f}")
            # print(f"Output/Input scale ratio: {output_scale/input_scale:.6f}")

            break
    
    return results

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)

def test_equivariance_multi_seeds(num_tests=10):
    """"""
    # 
    import random
    import time
    random.seed(int(time.time()))
    
    # 
    seeds = [random.randint(1, 100000) for _ in range(num_tests)]
    num_seeds = len(seeds)
    # seeds = [28461]
    
    # 
    all_results = []
    
    # 
    for i, seed in enumerate(seeds):
        print(f"\n[{i+1}/{num_seeds}] Testing with random seed: {seed}")
        results = test_equivariance_with_seed(seed)
        all_results.append(results)
        
        # 
        print(f"  Rotation invariance error: {results['rotation_invariance_error']:.8f}")
        print(f"  Rotation invariance error on h: {results['rotation_invariance_error_h']:.8f}")
        print(f"  Rotation equivariance error: {results['rotation_equivariance_error']:.8f}")
    
    # 
    stats = {metric: {} for metric in all_results[0].keys()}
    
    for metric in stats.keys():
        values = [result[metric] for result in all_results]
        stats[metric]['mean'] = sum(values) / len(values)
        stats[metric]['min'] = min(values)
        stats[metric]['max'] = max(values)
        stats[metric]['std'] = (sum((x - stats[metric]['mean'])**2 for x in values) / len(values))**0.5
    
    # 
    print("\n" + "="*60)
    print(f"RESULTS OVER {num_seeds} RANDOM SEEDS")
    print("="*60)
    
    metrics_display = {
        'rotation_invariance_error': 'Rotation invariance error (spatial)',
        'rotation_invariance_error_h': 'Rotation invariance error (non-spatial)',
        'rotation_equivariance_error': 'Rotation equivariance error'
    }
    
    for metric, display_name in metrics_display.items():
        print(f"\n{display_name}:")
        print(f"  Mean: {stats[metric]['mean']:.8f}")
        print(f"  Min:  {stats[metric]['min']:.8f}")
        print(f"  Max:  {stats[metric]['max']:.8f}")
        print(f"  Std:  {stats[metric]['std']:.8f}")
    
    return stats, all_results

if __name__ == "__main__":
    # 
    num_tests = 1
    stats, results = test_equivariance_multi_seeds(num_tests)
    