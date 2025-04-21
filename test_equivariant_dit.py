import torch
from sym_nn.sym_nn import DiTGaussian_dynamics
from equivariant_diffusion.utils import (
    assert_mean_zero_with_mask,
    remove_mean_with_mask,
    assert_correctly_masked
)
import sys
sys.path.append('..')
from qm9 import dataset
from types import SimpleNamespace
from configs.datasets_config import get_dataset_info

torch.random.manual_seed(1)
dtype = torch.float32

def random_rotation_matrix():
    """生成3D随机旋转矩阵"""
    theta = torch.randn(3)
    theta = theta / torch.norm(theta)
    K = torch.tensor([[0, -theta[2], theta[1]],
                    [theta[2], 0, -theta[0]],
                    [-theta[1], theta[0], 0]])
    return torch.matrix_exp(K)

def test_equivariance_with_seed(seed):
    torch.manual_seed(seed)
    
    # 模型配置
    device = 'cpu'
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
    
    # 数据加载
    dataloaders, _ = dataset.retrieve_dataloaders(cfg)
    loader = dataloaders['train']
    dataset_info = get_dataset_info('qm9', False)
    
    # 模型参数
    dynamics_in_node_nf = len(dataset_info['atom_decoder']) + int(False)
    model_params = {
        'args': cfg,
        'in_node_nf': dynamics_in_node_nf,
        'context_node_nf': 0,
        'xh_hidden_size': 64,
        'K': 4,
        'hidden_size': 256,
        'depth': 8,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'mlp_dropout': 0.1,
        'mlp_type': "swiglu",
        'n_dims': 3,
        'device': device
    }
    
    # 初始化模型
    model = DiTGaussian_dynamics(**model_params)
    model.eval()
    
    results = {}
    with torch.no_grad():
        for data in loader:
            # 数据准备
            x = data['positions'].to(device, dtype)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = data['charges'].to(device, dtype)
            
            # 预处理
            x = remove_mean_with_mask(x, node_mask)
            xh = torch.cat([x, one_hot, charges], dim=2)
            t = torch.randint(0, 1, (x.size(0), 1), device=device).float()
            
            # 原始输出
            out_orig = model._forward(t, xh, node_mask, edge_mask, context=None)
            
            # ========== 旋转测试 ==========
            weight = 1e6
            R = random_rotation_matrix()
            
            # 旋转输入
            xh_rot = xh.clone()
            xh_rot[..., :3] = torch.matmul(xh[..., :3], R.T)
            xh_rot = xh_rot * node_mask  # 应用掩码
            
            # 旋转后输出
            out_rot = model._forward(t, xh_rot, node_mask, edge_mask, context=None)
            
            # 计算误差指标
            spatial_diff = out_orig[..., :3] - out_rot[..., :3]
            results['rot_invariance'] = torch.mean((spatial_diff * weight)**2).item()
            
            # 旋转等变性检查
            orig_rotated = torch.matmul(out_orig[..., :3], R.T)
            equiv_diff = orig_rotated - out_rot[..., :3]
            results['rot_equivariance'] = torch.mean((equiv_diff * weight)**2).item()
            
            # ========== 平移测试 ==========
            translation = torch.randn(1, 3) * 5
            xh_trans = xh.clone()
            xh_trans[..., :3] += translation
            xh_trans = xh_trans * node_mask  # 应用掩码
            
            # 平移后输出
            out_trans = model._forward(t, xh_trans, node_mask, edge_mask, context=None)
            
            # 平移等变性检查
            trans_diff = out_orig[..., :3] - out_trans[..., :3]
            results['trans_equivariance'] = torch.mean((trans_diff * weight)**2).item()
            
            # ========== 特征不变性检查 ==========
            if out_orig.shape[-1] > 3:
                feature_diff = out_orig[..., 3:] - out_rot[..., 3:]
                results['feat_invariance'] = torch.mean((feature_diff * weight)**2).item()
            else:
                results['feat_invariance'] = 0.0
            
            # ========== 调试信息 ==========
            print(f"\n[Debug Info] Output ranges:")
            print(f"Spatial  : {out_orig[...,:3].min():.4f} ~ {out_orig[...,:3].max():.4f}")
            print(f"Features : {out_orig[...,3:].min():.4f} ~ {out_orig[...,3:].max():.4f}")
            print(f"Rot invariance error  : {results['rot_invariance']:.6f}")
            print(f"Rot equivariance error: {results['rot_equivariance']:.6f}")
            
            break  # 仅测试第一个batch
    
    return results

def test_equivariance_multi_seeds(num_tests=5):
    import random
    seeds = [random.randint(1, 10000) for _ in range(num_tests)]
    
    all_results = []
    for i, seed in enumerate(seeds):
        print(f"\n=== Test {i+1}/{num_tests} (Seed: {seed}) ===")
        results = test_equivariance_with_seed(seed)
        all_results.append(results)
        
    # 结果统计
    metrics = ['rot_invariance', 'rot_equivariance', 'trans_equivariance', 'feat_invariance']
    stats = {
        metric: {
            'mean': sum(r[metric] for r in all_results)/num_tests,
            'max': max(r[metric] for r in all_results),
            'min': min(r[metric] for r in all_results)
        } 
        for metric in metrics
    }
    
    print("\n=== Final Statistics ===")
    for metric in metrics:
        print(f"{metric:20} | Mean: {stats[metric]['mean']:.6f} | Range: [{stats[metric]['min']:.6f}, {stats[metric]['max']:.6f}]")
    
    return stats

if __name__ == "__main__":
    # 运行5次不同种子的测试
    stats = test_equivariance_multi_seeds(num_tests=1)