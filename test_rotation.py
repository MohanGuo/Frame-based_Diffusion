import torch
# from egnn.models_new import EGNN_dynamics_QM9
from model.transformer_dynamic import TransformerDynamics_2
# from model.transformer_dynamic_baseline_transformer import TransformerDynamics_2
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
# from utils import visualize_molecule, random_rotation
import sys
sys.path.append('..')
from qm9 import dataset
from types import SimpleNamespace
from configs.datasets_config import get_dataset_info
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# torch.set_default_dtype(torch.float64)
# dtype = torch.float64
torch.random.manual_seed(1)
dtype = torch.float

def random_rotation(x):
    bs, n_nodes, n_dims = x.size()
    device = x.device
    angle_range = np.pi * 2

    if n_dims == 3:
        # 使用轴角法避免万向节锁
        axis = torch.randn(bs, 3, device=device)
        axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)  # 单位化
        angles = torch.rand(bs, 1, device=device) * angle_range - np.pi
        
        # Rodrigues公式生成旋转矩阵
        K = torch.zeros(bs, 3, 3, device=device)
        K[:, 0, 1] = -axis[:, 2]
        K[:, 0, 2] = axis[:, 1]
        K[:, 1, 0] = axis[:, 2]
        K[:, 1, 2] = -axis[:, 0]
        K[:, 2, 0] = -axis[:, 1]
        K[:, 2, 1] = axis[:, 0]
        
        I = torch.eye(3, device=device).unsqueeze(0).repeat(bs, 1, 1)
        sin_theta = torch.sin(angles).unsqueeze(-1)
        cos_theta = (1 - torch.cos(angles)).unsqueeze(-1)
        R = I + sin_theta * K + cos_theta * (K @ K)
        
        x = x.transpose(1, 2)          # [bs, 3, n_nodes]
        x = torch.bmm(R, x)            # 
        x = x.transpose(1, 2)          # [bs, n_nodes, 3]
    else:
        # 2D  
        theta = torch.rand(bs, 1, 1).to(device) * angle_range - np.pi
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_row0 = torch.cat([cos_theta, -sin_theta], dim=2)
        R_row1 = torch.cat([sin_theta, cos_theta], dim=2)
        R = torch.cat([R_row0, R_row1], dim=1)

        x = x.transpose(1, 2)
        x = torch.matmul(R, x)
        x = x.transpose(1, 2)
    
    return x.contiguous()

def visualize_molecule(x, node_mask, batch_index=0, save_path='molecule.png'):
    """增强版分子可视化"""
    coords = x[batch_index].detach().cpu().numpy()
    mask = node_mask[batch_index].squeeze().detach().cpu().numpy()
    valid_mask = mask > 0
    valid_coords = coords[valid_mask]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制原子（根据Z坐标着色）
    sc = ax.scatter(
        valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2],
        c=valid_coords[:, 2], cmap='viridis', s=100, depthshade=True,
        edgecolors='k', linewidth=0.5
    )
    
    # 添加颜色条
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label('Z Coordinate', rotation=270, labelpad=15)
    
    # 设置等比例轴
    max_range = np.array([
        valid_coords[:,0].max()-valid_coords[:,0].min(), 
        valid_coords[:,1].max()-valid_coords[:,1].min(),
        valid_coords[:,2].max()-valid_coords[:,2].min()
    ]).max() * 0.5
    
    mid_x = (valid_coords[:,0].max()+valid_coords[:,0].min()) * 0.5
    mid_y = (valid_coords[:,1].max()+valid_coords[:,1].min()) * 0.5
    mid_z = (valid_coords[:,2].max()+valid_coords[:,2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # 设置视角
    ax.view_init(elev=30, azim=45)  # 固定视角
    
    # 添加连接线（示例：显示所有原子间的连接）
    for i in range(len(valid_coords)):
        for j in range(i+1, len(valid_coords)):
            ax.plot(
                [valid_coords[i,0], valid_coords[j,0]],
                [valid_coords[i,1], valid_coords[j,1]], 
                [valid_coords[i,2], valid_coords[j,2]],
                'gray', alpha=0.3, linewidth=0.5
            )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def test_equivariance_with_seed(seed):
    """运行单次等变性测试并返回指定的误差指标"""
    # 设置随机种子
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
    