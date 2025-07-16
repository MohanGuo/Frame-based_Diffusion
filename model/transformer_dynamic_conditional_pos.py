import torch
import torch.nn as nn
import math
import numpy as np
# from .transformer import Transformer
from .transformer_conditional_pos import DiT
import sys
sys.path.append('..')
from equivariant_diffusion.utils import remove_mean, remove_mean_with_mask, assert_mean_zero_with_mask
from egnn.models import EGNN_dynamics_QM9_MC
from egnn.egnn_mc import EGNN as EGNN_mc
import time
from torch_geometric.nn import global_mean_pool, global_add_pool
from sym_nn.utils import GaussianLayer
EPS = 1e-8

class TransformerDynamics_2(nn.Module):
    def __init__(self, args, egnn, in_node_nf, context_node_nf, n_dims, 
                 hidden_nf=64, device='cpu', n_heads=4, 
                 n_layers=4, condition_time=True, debug=False):

        super().__init__()
        in_node_nf -= 1
        self.in_node_nf = in_node_nf
        self.context_node_nf = context_node_nf
        self.device = device
        self.n_dims = n_dims
        self._edges_dict = {}
        self.condition_time = condition_time

        self.egnn = egnn

        
        # self.feature_embedding = nn.Sequential(
        #     nn.Linear(in_node_nf, hidden_nf),
        #     nn.SiLU(),
        #     nn.Linear(hidden_nf, 2)
        # )

        #DiT init
        # self.gaussian_embedder = GaussianLayer(K=184)
        xh_hidden_size = 184
        K = 184
        hidden_size = 384
        depth = 12
        num_heads = 6
        mlp_ratio = 4.0
        mlp_dropout = 0.0
        mlp_type = "swiglu"
        # self.xh_embedder = nn.Linear(n_dims+in_node_nf+context_node_nf, xh_hidden_size)
        # self.pos_embedder = nn.Linear(K, hidden_size-xh_hidden_size)

        # 定义共享参数
        self.num_dit_modules = 3
        self.hidden_size = 384
        self.xh_hidden_size = 184
        K = 184  # 高斯嵌入维度
        
        # 删除全局的 self.gaussian_embedder, self.pos_embedder, self.feature_embedding
        
        # 为每个 DiT 模块独立定义组件
        self.dit_layers = nn.ModuleList()
        for _ in range(self.num_dit_modules):
            # === 独立定义每个模块的组件 ===
            # (1) 高斯嵌入层
            gaussian_embedder = GaussianLayer(K=K)
            # (2) 位置编码投影层
            pos_embedder = nn.Linear(K, self.hidden_size - self.xh_hidden_size)
            # (3) 特征嵌入层（用于边特征）
            feature_embedding = nn.Sequential(
                nn.Linear(in_node_nf, hidden_nf),
                nn.SiLU(),
                nn.Linear(hidden_nf, 2)
            )
            xh_embedder = nn.Linear(n_dims+in_node_nf+context_node_nf, xh_hidden_size)
            # (4) DiT 主模块
            dit = DiT(
                out_channels=n_dims + in_node_nf + context_node_nf,
                hidden_size=self.hidden_size,
                depth=4,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                mlp_dropout=mlp_dropout,
                mlp_type=mlp_type,
                use_fused_attn=True,
                x_emb="identity",
                edge_dim=8,
                d_model=hidden_nf
            )
            
            # 附加独立组件到 DiT 模块
            dit.gaussian_embedder = gaussian_embedder
            dit.pos_embedder = pos_embedder
            dit.feature_embedding = feature_embedding
            dit.xh_embedder = xh_embedder
            
            # 初始化权重
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            dit.gaussian_embedder.apply(_basic_init)
            dit.pos_embedder.apply(_basic_init)
            dit.feature_embedding.apply(_basic_init)
            dit.xh_embedder.apply(_basic_init)
            
            self.dit_layers.append(dit)

    def _forward(self, t, xh, node_mask, edge_mask, context):
        # start_1 = time.time()
        bs, n_nodes, dims = xh.shape
        h_dims = dims - self.n_dims
        edges = self.get_adj_matrix(n_nodes, bs, self.device)
        edges = [x.to(self.device) for x in edges]

        x_input =remove_mean_with_mask(xh[..., :self.n_dims], node_mask)
        x_input = x_input.view(bs*n_nodes, -1)
        
        xh = xh.view(bs*n_nodes, -1)#.clone() * node_mask

        h = xh[:, self.n_dims:].clone()
        
        h_egnn = h.clone()

        
        node_mask = node_mask.view(bs*n_nodes, 1)
        edge_mask = edge_mask.view(bs*n_nodes*n_nodes, 1)
        

        ################## MG: pass egnn to get invariant coordinates ###################
        ##Multichannel
        output, vel_inverse, pose = self.egnn._forward(h_egnn, x_input, edges, node_mask, edge_mask, context=None, bs=bs, n_nodes=n_nodes, dims=dims)
        x_invariant = output[..., :self.n_dims]
        # print(f"x_invariant: {x_invariant}")
        # x_invariant = x_invariant.view(bs*n_nodes, -1)
        x_invariant = x_invariant.view(bs, n_nodes, -1)
        x_invariant = remove_mean_with_mask(x_invariant, node_mask.view(bs, n_nodes, 1))
        #################################################################################
        
        # ######edge bias#####
        # x_invariant_edge = x_invariant.view(bs*n_nodes, -1)
        # radial, coord_diff = coord2diff(x_invariant_edge, edges)
        # coord_diff = coord_diff.view(bs, n_nodes, n_nodes, 3)
        # radial = radial.view(bs, n_nodes, n_nodes, 1)
        # #add h into edge features
        # h_embedded = self.feature_embedding(h)
        # rows, cols = edges
        # h_i = h_embedded[rows]
        # h_j = h_embedded[cols]
        # node_edge_features = torch.cat([h_i, h_j], dim=1)
        # node_features_matrix = node_edge_features.view(bs, n_nodes, n_nodes, -1)
        # edge_features = torch.cat([radial, coord_diff, node_features_matrix], dim=-1)


        ######DiT input#####
        # x = xh[:, 0:self.n_dims].clone()
        # x = x.view(bs, n_nodes, 3)
        node_mask = node_mask.view(bs, n_nodes, 1)
        # x = remove_mean_with_mask(x, node_mask)
        # N = torch.sum(node_mask, dim=1, keepdims=True)  # [bs, 1, 1]
        # pos_emb = self.gaussian_embedder(x, node_mask)  # [bs, n_nodes, n_nodes, K]
        # pos_emb = torch.sum(self.pos_embedder(pos_emb), dim=-2) / N  # [bs, n_nodes, K]
        # xh = torch.cat([x.clone(), h.view(bs, n_nodes, -1)], dim=-1)
        # xh = torch.cat([x_invariant.clone(), h.view(bs, n_nodes, -1)], dim=-1)
        # xh = xh.view(bs, n_nodes, -1)
        # print(f"xh: {xh.shape}")
        # xh = self.xh_embedder(xh)
        # print(f"xh: {xh}")
        # xh = node_mask * torch.cat([xh, pos_emb], dim=-1)

        ######DiT##########
        # 初始 DiT 输入准备
        # 初始化坐标和原子特征
        current_coord = x_invariant  # 初始坐标
        current_h = h.view(bs, n_nodes, -1).clone()  # 原子特征

        # 逐层处理
        for i, dit_layer in enumerate(self.dit_layers):
            # === 步骤1: 生成当前层的 edge_features 和 pos_emb ===
            # (1) 计算 edge_features
            x_invariant_edge = current_coord.view(bs * n_nodes, -1)
            radial, coord_diff = coord2diff(x_invariant_edge, edges)
            coord_diff = coord_diff.view(bs, n_nodes, n_nodes, 3)
            radial = radial.view(bs, n_nodes, n_nodes, 1)
            h_embedded = dit_layer.feature_embedding(current_h.view(bs * n_nodes, -1))  # 使用当前模块的 feature_embedding
            rows, cols = edges
            h_i = h_embedded[rows]
            h_j = h_embedded[cols]
            node_edge_features = torch.cat([h_i, h_j], dim=1).view(bs, n_nodes, n_nodes, -1)
            edge_features = torch.cat([radial, coord_diff, node_edge_features], dim=-1)
            
            # (2) 生成 pos_emb
            pos_emb = dit_layer.gaussian_embedder(current_coord, node_mask.view(bs, n_nodes, 1))
            pos_emb = torch.sum(
                dit_layer.pos_embedder(pos_emb), 
                dim=-2
            ) / torch.sum(node_mask.view(bs, n_nodes, 1), dim=1, keepdim=True)

            # === 步骤2: 准备输入并处理 ===
            # (1) 拼接输入特征（坐标 + 原子特征）并通过 xh_embedder
            xh_input = torch.cat([current_coord, current_h], dim=-1)
            xh_embedded = dit_layer.xh_embedder(xh_input)  # 使用当前模块的 xh_embedder
            # print(f"xh_embedded: {xh_embedded}")
            # (2) 拼接位置编码
            xh_input_full = torch.cat([xh_embedded, pos_emb], dim=-1)
            
            # (3) DiT处理
            xh_output = node_mask * dit_layer(
                t.squeeze(-1), 
                xh_input_full, 
                edge_features, 
                node_mask=node_mask.view(bs, n_nodes, 1), 
                batch_size=bs, 
                n_nodes=n_nodes
            )
            
            # === 步骤3: 更新坐标和特征 ===
            # (1) 提取新坐标（假设前n_dims维是坐标）
            new_coord = xh_output[:, :, :self.n_dims]
            # new_coord = remove_mean_with_mask(new_coord, node_mask.view(bs, n_nodes, 1))
            
            # (2) 更新原子特征（剩余维度）
            current_h = xh_output[:, :, self.n_dims:]
            
            # (3) 传递到下一层
            current_coord = new_coord
            # print(f"current_coord: {current_coord}")
        ####### Output ########
        x_final = current_coord
        h_final = current_h
        # print(f"x_final: {x_final.shape}")
        x_final = remove_mean_with_mask(x_final, node_mask)
        assert_mean_zero_with_mask(x_final, node_mask)
        x_final = x_final.view(bs * n_nodes, -1)
        h_final = h_final.view(bs * n_nodes, -1)
        # print(f"x_final: {x_final}")
        
        ################# if transform back #####################
        x_final_equivariant = torch.bmm(x_final.unsqueeze(1), pose).squeeze(1)
        # vel = (x_final_equivariant) * node_mask
        x_final_equivariant = x_final_equivariant.view(bs, n_nodes, -1)  # 恢复bs维度
        x_final_equivariant = remove_mean_with_mask(x_final_equivariant, node_mask)
        vel = (x_final_equivariant)
        #########################################################


        if torch.any(torch.isnan(vel)):
            print('Warning: detected nan, resetting transformer output to zero.')
            vel = torch.zeros_like(vel)

        if node_mask is None:
            vel = remove_mean(vel)
            print("No mask!")
        else:
            vel = remove_mean_with_mask(vel, node_mask.view(bs, n_nodes, 1))
            # print(f"Remove mean with mask!")

        # print(f"Shape of node_mask: {node_mask}")
        assert_mean_zero_with_mask(vel, node_mask.view(bs, n_nodes, 1))

        # start_5 = time.time()
        # print(f'One forward took {start_5 - start_1:.2f} seconds')
        h_final = h_final.view(bs, n_nodes, -1)
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

def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff