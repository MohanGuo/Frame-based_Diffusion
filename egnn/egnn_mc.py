from torch import nn
import torch

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_sum_vec(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1), data.size(2))
    segment_ids = segment_ids.unsqueeze(-1).unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, data.size(1), data.size(2))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    result.scatter_add_(0, segment_ids, data)
    return result 

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1), data.size(2))
    segment_ids = segment_ids.unsqueeze(-1).unsqueeze(-1)
    segment_ids = segment_ids.expand(-1, data.size(1), data.size(2))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_edge_nf, hidden_node_nf, hidden_coord_nf,edges_in_d=0,
                nodes_att_dim=0, act_fn=nn.ReLU(),recurrent=True, coords_weight=1.0,
                attention=False, clamp=False, norm_diff=False, tanh=False,
                num_vectors_in=1, num_vectors_out=1, last_layer=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        self.num_vectors_in = num_vectors_in
        self.num_vectors_out = num_vectors_out
        self.last_layer = last_layer
        edge_coords_nf = 1

        # input dimensions: 2 * input_nf + num_vectors_in + edges_in_d, edges_in_d represents the radial
        # The input is h[src], h[dst] and radial. Every channel has a radial.
        # print(f"input_edge: {input_edge}, num_vectors_in: {num_vectors_in}, edges_in_d: {edges_in_d}")
        # print(f"hidden_edge_nf: {hidden_edge_nf}, hidden_node_nf: {hidden_node_nf}, hidden_coord_nf: {hidden_coord_nf}")
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + num_vectors_in + edges_in_d, hidden_edge_nf),
            act_fn,
            nn.Linear(hidden_edge_nf, hidden_edge_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_edge_nf + input_nf + nodes_att_dim, hidden_node_nf),
            act_fn,
            nn.Linear(hidden_node_nf, output_nf))

        # the ourput dimension is num_vectors_in * num_vectors_out instead of 1
        layer = nn.Linear(hidden_coord_nf, num_vectors_in * num_vectors_out, bias=False)
        # torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        weight = layer.weight.data.view(num_vectors_out, num_vectors_in, -1)
        hidden_dim = weight.shape[2]
        
        # 方法1: 使用显著不同的增益值并添加方向性偏好
        for i in range(num_vectors_out):
            # 使用指数增长的增益值，差异会更明显
            gain = 0.05 * (2 ** i)  # 0.05, 0.1, 0.2, 0.4, 0.8, ...
            
            # 对每个输出通道单独初始化
            torch.nn.init.xavier_uniform_(weight[i], gain=gain)
            
            # 为每个通道添加不同的方向性偏好
            # 为奇数通道添加正向偏移，偶数通道添加负向偏移
            bias_direction = 0.5 * (-1 if i % 2 == 0 else 1)
            weight[i] += bias_direction


        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_edge_nf, hidden_coord_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_edge_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, radial, edge_attr):
        # print(f"edge_attr: {edge_attr}")
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        # print(f"Shape of input: {out.shape}")
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, radial, edge_feat):
        row, col = edge_index
        # print(f"Shape of output of coord_mlp: {self.coord_mlp(edge_feat).shape}")
        coord_matrix = self.coord_mlp(edge_feat).view(-1, self.num_vectors_in, self.num_vectors_out)
        if coord_diff.dim() == 2:
            coord_diff = coord_diff.unsqueeze(2)
            coord = coord.unsqueeze(2).repeat(1, 1, self.num_vectors_out)
        # coord_diff = coord_diff / radial.unsqueeze(1)
        trans = torch.einsum('bij,bci->bcj', coord_matrix, coord_diff)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        if self.last_layer:
            coord = coord.mean(dim=2, keepdim=True) + agg * self.coords_weight
        else:
            coord += agg * self.coords_weight
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        if radial.dim() == 3:
            radial = radial.squeeze(1)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, radial, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr




class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_edge_nf, hidden_node_nf, hidden_coord_nf,
                edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(),
                recurrent=True, coords_weight=1.0, attention=False,
                num_vectors_in=1, num_vectors_out=1,
                update_coords=False, last_layer=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_edge_nf, hidden_node_nf, hidden_coord_nf,
                edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim,
                act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight,
                attention=attention, num_vectors_in=num_vectors_in, num_vectors_out=num_vectors_out,
                last_layer=last_layer)
        self.update_coords = update_coords
        if not self.update_coords:
            del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        coord_matrix = self.coord_mlp(edge_feat).view(-1, self.num_vectors_in, self.num_vectors_out) # [batch_size, num_vectors_in, num_vectors_out]
        # print(f"Shape of coord_matrix: {coord_matrix}")
        # Shape of coord_matrix: torch.Size([16928, 1, 7])
        # Shape of coord_matrix: torch.Size([16928, 7, 7])
        # Shape of coord_matrix: torch.Size([16928, 7, 7])
        # Shape of coord_matrix: torch.Size([16928, 7, 7])
        # Shape of coord_matrix: torch.Size([16928, 7, 7])
        # Shape of coord_matrix: torch.Size([16928, 7, 3])

        # if self.last_layer:
        #     print(f"coord_matrix: {coord_matrix[0, ...]}")

        # print(f"Shape of coord_diff: {coord_diff.shape}")
        if coord_diff.dim() == 2:
            coord_diff = coord_diff.unsqueeze(2)
            coord = coord.unsqueeze(2).repeat(1, 1, self.num_vectors_out)
        # print(f"Shape of coord_diff after: {coord_diff.shape}")
        # coord_diff = coord_diff / radial.unsqueeze(1)
        trans = torch.einsum('bij,bci->bcj', coord_matrix, coord_diff)
        # trans[b,c,j] = sum_i(coord_matrix[b,i,j] * coord_diff[b,c,i])
        # why not use edge mask? Shape changed. Is there a problem?
        trans = trans# * edge_mask
        # print(f"Shape of trans: {trans.shape}")
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        # print(f"Shape of agg: {agg.shape}")
        if self.last_layer:
            coord = coord.mean(dim=2, keepdim=True) + agg * self.coords_weight
        else:
            coord += agg * self.coords_weight
        # print(f"Shape of output of coord_model: {coord.shape}")
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask
        # Modified to include coordinates
        # the input into the coord model is the edge_feat, output of the edge model.
        if self.update_coords:
            coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        # coord is the updated coordinates
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # print(f"Shape of h in E_GCL_mask: {h.shape}")
        # print(f"Shape of coord in E_GCL_mask: {coord.shape}")
        # print(f"Shape of node_mask in E_GCL_mask: {node_mask.shape}")
        # h = h * node_mask
        # node_mask_unsqueeze = node_mask.unsqueeze(-1)
        # coord = coord * node_mask_unsqueeze
        

        return h, coord, edge_attr



class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_edge_nf, hidden_node_nf, hidden_coord_nf,
                device='cpu', act_fn=nn.SiLU(), n_layers=4,
                coords_weight=1.0,attention=False, node_attr=1,
                num_vectors=1, update_coords=False, num_vectors_out=1):
        super(EGNN, self).__init__()
        self.hidden_edge_nf = hidden_edge_nf
        self.hidden_node_nf = hidden_node_nf
        self.hidden_coord_nf = hidden_coord_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_node_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0

        # in_edge_nf = 0 in default
        self.add_module("gcl_%d" % 0,
                E_GCL_mask(self.hidden_node_nf, self.hidden_node_nf,
                    self.hidden_edge_nf, self.hidden_node_nf, self.hidden_coord_nf,
                    edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr,
                    act_fn=act_fn, recurrent=True,
                    coords_weight=coords_weight, attention=attention,
                    num_vectors_in=1, num_vectors_out=num_vectors, update_coords=update_coords))
        for i in range(1, n_layers - 1):
            self.add_module("gcl_%d" % i,
                E_GCL_mask(self.hidden_node_nf, self.hidden_node_nf,
                    self.hidden_edge_nf, self.hidden_node_nf, self.hidden_coord_nf,
                    edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr,
                    act_fn=act_fn, recurrent=True,
                    coords_weight=coords_weight, attention=attention,
                    num_vectors_in=num_vectors, num_vectors_out=num_vectors, update_coords=update_coords))
        self.add_module("gcl_%d" %  (n_layers - 1),
            E_GCL_mask(self.hidden_node_nf, self.hidden_node_nf,
                self.hidden_edge_nf, self.hidden_node_nf, self.hidden_coord_nf,
                edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr,
                act_fn=act_fn, recurrent=True,
                coords_weight=coords_weight, attention=attention,
                num_vectors_in=num_vectors, num_vectors_out=num_vectors_out,
                update_coords=update_coords, last_layer=True))


        self.node_dec = nn.Sequential(nn.Linear(self.hidden_node_nf, self.hidden_node_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_node_nf, self.hidden_node_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_node_nf, self.hidden_node_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_node_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        # print(f"Shape of h in stage 1: {h.shape}")
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, x, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, x, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)
        # print(f"Shape of h in stage 2: {h.shape}")
        # h = self.node_dec(h)
        # print(f"Shape of h in stage 3: {h.shape}")
        # print(f"Shape of x: {x.shape}")
        # h = h * node_mask
        # h = h.view(-1, n_nodes, self.hidden_node_nf)
        # h = torch.sum(h, dim=1)
        # pred = self.graph_dec(h)
        # return pred.squeeze(1)
        # print
        return h, x
    

## Testing
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

def test_egnn():
    in_node_nf = 5        
    in_edge_nf = 2        
    hidden_edge_nf = 32   
    hidden_node_nf = 64   
    hidden_coord_nf = 32  
    n_layers = 3         
    
    model = EGNN(
        in_node_nf=in_node_nf,
        in_edge_nf=in_edge_nf,
        hidden_edge_nf=hidden_edge_nf,
        hidden_node_nf=hidden_node_nf,
        hidden_coord_nf=hidden_coord_nf,
        n_layers=n_layers,
        update_coords=True,
        attention=True,
        num_vectors=7,
        num_vectors_out=9
    )
    
    batch_size = 2
    n_nodes = 4
    
    h0 = torch.randn(batch_size * n_nodes, in_node_nf)
    x = torch.randn(batch_size * n_nodes, 3)
    
    edges = []
    for b in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edges.append([b * n_nodes + i, b * n_nodes + j])
    edges = torch.tensor(edges).t().long()
    
    edge_attr = torch.randn(edges.size(1), in_edge_nf)
    node_mask = torch.ones(batch_size * n_nodes, 1)
    edge_mask = torch.ones(edges.size(1), 1)
    
    print("Input shapes:")
    print(f"Node features: {h0.shape}")
    print(f"Coordinates: {x.shape}")
    print(f"Edges: {edges.shape}")
    print(f"Edge attributes: {edge_attr.shape}")
    
    try:
        _, output = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
        print("\nSuccess! Output shape:", output.shape)
        # print("Output values:", output)
    except Exception as e:
        print("\nError during forward pass:", str(e))
        raise e
    
def test_egnn_equivariance():
    print("\nTesting EGNN Equivariance Properties...")
    
    # Initialize model with same parameters
    in_node_nf = 5        
    in_edge_nf = 2        
    hidden_edge_nf = 32   
    hidden_node_nf = 64   
    hidden_coord_nf = 32  
    n_layers = 3         
    
    model = EGNN(
        in_node_nf=in_node_nf,
        in_edge_nf=in_edge_nf,
        hidden_edge_nf=hidden_edge_nf,
        hidden_node_nf=hidden_node_nf,
        hidden_coord_nf=hidden_coord_nf,
        n_layers=n_layers,
        update_coords=True,
        attention=True,
        num_vectors=7,
        num_vectors_out=4
    )
    
    # Fix random seed for reproducibility
    torch.manual_seed(42)
    
    # Create input data
    batch_size = 1  # Using batch_size=1 for simplicity
    n_nodes = 4
    
    h0 = torch.randn(batch_size * n_nodes, in_node_nf)
    x = torch.randn(batch_size * n_nodes, 3)
    
    # Create edges (fully connected graph)
    edges = []
    for b in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edges.append([b * n_nodes + i, b * n_nodes + j])
    edges = torch.tensor(edges).t().long()
    
    edge_attr = torch.randn(edges.size(1), in_edge_nf)
    node_mask = torch.ones(batch_size * n_nodes, 1)
    edge_mask = torch.ones(edges.size(1), 1)
    
    # Get original output
    _, coord_output1 = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
    
    # # Test translation equivariance
    # print(f"Shape of x: {x.shape}")
    # translation = torch.tensor([1.0, 2.0, -0.5]).reshape(1, 3)  # Add extra dimension
    # translation = translation.expand(-1, coord_output1.size(1), -1)  # Expand to match coord_output1's size
    # x_translated = x + translation
    
    # _, coord_output_translated = model(h0, x_translated, edges, edge_attr, node_mask, edge_mask, n_nodes)
    
    # # For translation test
    # translation_error = torch.norm(coord_output_translated[..., :3] - (coord_output1[..., :3] + translation))
    # print(f"\nTranslation Equivariance Error: {translation_error.item():.6f}")
    translation = torch.tensor([1.0, 2.0, -0.5]).reshape(1, 3, 1)  # [1, 3, 1] to match coordinate dimension
    x_translated = x + translation.squeeze(-1)  # squeeze to match input x shape
    
    _, coord_output_translated = model(h0, x_translated, edges, edge_attr, node_mask, edge_mask, n_nodes)
    
    # Apply translation along the coordinate dimension (dim=1)
    translation_error = torch.norm(coord_output_translated - (coord_output1 + translation))
    print(f"\nTranslation Equivariance Error: {translation_error.item():.6f}")
    




    
    
    # Test rotation equivariance
    # Create a random rotation matrix using SVD
    random_matrix = torch.randn(3, 3)
    U, _, V = torch.svd(random_matrix)
    rotation_matrix = torch.mm(U, V.t())
    
    # Ensure it's a proper rotation matrix (determinant = 1)
    if torch.det(rotation_matrix) < 0:
        V[:, -1] = -V[:, -1]
        rotation_matrix = torch.mm(U, V.t())
    
    x_rotated = torch.mm(x, rotation_matrix.t())
    
    _, coord_output_rotated = model(h0, x_rotated, edges, edge_attr, node_mask, edge_mask, n_nodes)
    
    # Compare with rotated output
    print(f"Shape of coord_output1: {coord_output1.shape}")
    rotated_coord_output1 = torch.einsum('njs,ji->nis', coord_output1, rotation_matrix.T)
    rotation_error = torch.norm(coord_output_rotated - rotated_coord_output1)
    # rotation_error = torch.norm(coord_output_rotated[:, :, 1] - torch.matmul(coord_output1[:, :, 1], rotation_matrix.t()))
    print(f"Rotation Equivariance Error: {rotation_error.item():.6f}")
    
    # Test scale equivariance
    # scale = 2.0
    # x_scaled = x * scale
    
    # _, coord_output_scaled = model(h0, x_scaled, edges, edge_attr, node_mask, edge_mask, n_nodes)
    
    # scale_error = torch.norm(coord_output_scaled - coord_output1 * scale)
    # print(f"Scale Equivariance Error: {scale_error.item():.6f}")

def test_egnn_equivariance_2():
    print("\nTesting EGNN Equivariance Properties...")
    
    # Initialize model with same parameters
    in_node_nf = 5        
    in_edge_nf = 2        
    hidden_edge_nf = 32   
    hidden_node_nf = 64   
    hidden_coord_nf = 32  
    n_layers = 3         
    
    model = EGNN(
        in_node_nf=in_node_nf,
        in_edge_nf=in_edge_nf,
        hidden_edge_nf=hidden_edge_nf,
        hidden_node_nf=hidden_node_nf,
        hidden_coord_nf=hidden_coord_nf,
        n_layers=n_layers,
        update_coords=True,
        attention=True,
        num_vectors=7,
        num_vectors_out=3
    )
    
    # Fix random seed for reproducibility
    torch.manual_seed(100)
    
    # Create input data
    batch_size = 32
    n_nodes = 19
    
    h0 = torch.randn(batch_size * n_nodes, in_node_nf)
    x = torch.randn(batch_size * n_nodes, 3)
    
    # Create random node mask first
    node_mask = torch.bernoulli(torch.ones(batch_size * n_nodes, 1) * 0.8)  # 80% probability of being 1
    # Ensure at least one node is active
    if node_mask.sum() == 0:
        node_mask[0] = 1
    
    # Create edges and edge mask based on node mask
    edges = []
    edge_mask = []
    for b in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edges.append([b * n_nodes + i, b * n_nodes + j])
                    # Edge is active only if both nodes are active
                    is_edge_active = (node_mask[b * n_nodes + i] and 
                                    node_mask[b * n_nodes + j]).float()
                    edge_mask.append(is_edge_active)
    
    edges = torch.tensor(edges).t().long()
    edge_mask = torch.tensor(edge_mask).reshape(-1, 1)
    edge_attr = torch.randn(edges.size(1), in_edge_nf)
    
    print(f"Active nodes: {node_mask.sum().item()}/{len(node_mask)}")
    print(f"Active edges: {edge_mask.sum().item()}/{len(edge_mask)}")
    
    # Get original output
    h1, coord_output1 = model(h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes)
    
    # Test translation equivariance
    translation = torch.tensor([1.0, 2.0, -0.5]).reshape(1, 3, 1)
    x_translated = x + translation.squeeze(-1)
    
    h2, coord_output_translated = model(h0, x_translated, edges, edge_attr, node_mask, edge_mask, n_nodes)
    
    # Apply translation along the coordinate dimension (dim=1)
    translation_error = torch.norm((coord_output_translated.permute(0, 2, 1) - (coord_output1.permute(0, 2, 1) + translation)))# * node_mask.unsqueeze(-1))
    print(f"\nTranslation Equivariance Error (with masks): {translation_error.item():.6f}")

    translation_error = torch.norm((h2 - (h1)))# * node_mask.unsqueeze(-1))
    print(f"\nTranslation Equivariance Error (with masks): {translation_error.item():.6f}")
    


if __name__ == "__main__":
    # test_egnn()
    test_egnn_equivariance()
    # test_egnn_equivariance_2()