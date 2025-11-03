import torch
# from egnn.models_new import EGNN_dynamics_QM9
from model.transformer_dynamic import TransformerDynamics_2
# from model.transformer_dynamic_baseline_transformer import TransformerDynamics_2
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
from utils import visualize_molecule, random_rotation
import sys
sys.path.append('..')
from qm9 import dataset
from types import SimpleNamespace
from configs.datasets_config import get_dataset_info

# torch.set_default_dtype(torch.float64)
# dtype = torch.float64
torch.random.manual_seed(1)
dtype = torch.float

def test_equivariance_with_seed(seed):
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
    
    results = {
        'rotation_invariance_error': 0.0,
        'rotation_invariance_error_h': 0.0,
        'rotation_equivariance_error': 0.0
    }

    
    with torch.no_grad():
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
            
            print(f"Shape of x in test rotation: {x.shape}")
            print(f"Shape of node_mask in test rotation: {node_mask.shape}")

            visualize_molecule(x, node_mask, batch_index=0, save_path='molecule_train.png')
            x_rot = random_rotation(x)
            visualize_molecule(x_rot, node_mask, batch_index=0, save_path='molecule_train_rotated.png')
            
            break
    
    return results

def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


if __name__ == "__main__":
    test_equivariance_with_seed(0)
    