import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # d_model: embedding dimension of every token
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads

        # d_k is the dimension of each head
        self.d_k = d_model // num_heads
        
        # Q, K, V, Output are linear layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # For masked attention
        if mask is not None:
            
            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, self.num_heads, -1, -1)
            # (batch_size, num_heads, num_nodes, num_nodes)
            # print(f"Shape of mask: {mask.shape}")
            # print(f"Shape of attn_scores: {attn_scores.shape}")
            # print(f"Mask 1: {mask[0,0,...]}")
            # print(f"Attention scores: {attn_scores}")
            attn_scores = attn_scores.masked_fill(mask == 0, 0)

        # Softmax along the last dimension
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, num_nodes, d_model = x.size()
        # After transpose: (batch_size, num_heads, num_nodes, d_k)
        return x.view(batch_size, num_nodes, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, num_nodes, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        # print(f"Shape of input multi-head: {Q.shape}")
        # [bs, nodes, features]
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # print(f"Mask: {mask}")
        # print(f"Q: {Q[0]}")
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        # 1. Self-Attention
        attn_output = self.self_attn(x, x, x, mask)
        # print(f"Shape of x: {x.shape}")
        #x: [bs, nodes, d_model]
        # print(f"attn_output: {attn_output}")
        
        # 2. Residual + Norm
        x = self.norm1(x + self.dropout(attn_output))

        # 3. Feed Forward
        ff_output = self.feed_forward(x)

        # 4. Residual + Norm
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # x is the output
        # 1. Self-Attention
        attn_output = self.self_attn(x, x, x, tgt_mask)
        
        # 2. Residual + Norm
        x = self.norm1(x + self.dropout(attn_output))

        # 3. Cross-Attention
        # enc_output is the encoder output
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)

        # 4. Residual + Norm
        x = self.norm2(x + self.dropout(attn_output))

        # 5. Feed Forward
        ff_output = self.feed_forward(x)

        # 6. Residual + Norm
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Transformer(nn.Module):
    def __init__(self, args, in_node_nf, device, d_model, num_heads, num_layers, d_ff, dropout):
        super(Transformer, self).__init__()
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        # Embedding layers for h and x
        self.h_embedding = nn.Linear(in_node_nf, d_model)
        self.x_embedding = nn.Linear(3, d_model)  # 3 = 3

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model * 2, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Output heads for noise prediction
        self.noise_h_head = nn.Linear(d_model * 2, in_node_nf)  # Noise for h (categorical + integer)
        # self.noise_x_head = nn.Linear(d_model * 2, 3)  # Noise for x (3D coordinates)
        self.noise_x_head = nn.Sequential(
        nn.Linear(d_model * 2, d_model),
        nn.SiLU(),
        nn.Linear(d_model, d_model // 2),
        nn.SiLU(),
        nn.Linear(d_model // 2, 3)
        )       

        self.to(device)

        # self.batch_size = args.batch_size

    # def generate_mask(self, h, x):
    #     # Generate masks if needed (e.g., for padding)
    #     # Here we assume no padding, so masks are not used
    #     src_mask = None
    #     return src_mask
    def generate_mask(self, h, x, node_mask=None, batch_size=0):
        """
        Input:
            node_mask: shape (batch_size*n_nodes, 1)
        Output:
            attention_mask: shape (batch_size, n_nodes, n_nodes)
        """
        if node_mask is None:
            return None
            
        # Calculate n_nodes from h or x shape
        # batch_size = self.batch_size  # We need to know batch_size beforehand
        total_nodes = node_mask.shape[0]
        n_nodes = total_nodes // batch_size
        
        # Reshape node_mask from (batch_size*n_nodes, 1) to (batch_size, n_nodes, 1)

        node_mask_i = node_mask.view(batch_size, n_nodes, 1)
        node_mask_j = node_mask.view(batch_size, 1, n_nodes)
        attention_mask = node_mask_i * node_mask_j

        return attention_mask

    def forward(self, h, x, edges=None, node_mask=None, edge_mask=None, batch_size=0):
        # h: (bs * n_nodes, in_node_nf)
        # x: (bs * n_nodes, 3)  # 3D coordinates
        bs_n_nodes = x.size(0)
        # print(f"Shape of input transformer: {x.shape}")

        # Embed h and x
        # h_embedded = self.h_embedding(h)  # (bs * n_nodes, d_model)
        # x_embedded = self.x_embedding(x)  # (bs * n_nodes, d_model)
        h_embedded = self.dropout(self.h_embedding(h))
        x_embedded = self.dropout(self.x_embedding(x))

        # Concatenate h and x embeddings
        xh_embedded = torch.cat([h_embedded, x_embedded], dim=-1)  # (bs * n_nodes, d_model * 2)

        # Reshape for Transformer input: (bs, n_nodes, d_model * 2)
        xh_embedded = xh_embedded.view(batch_size, bs_n_nodes // batch_size, self.d_model * 2)

        # Generate masks (if needed)
        # print(f"Node mask shape: {node_mask.shape}")
        attn_mask = self.generate_mask(h, x, node_mask, batch_size)
        # attn_mask = None

        # print(f"Attn mask shape: {attn_mask.shape}")
        # print(f"XH embedded shape: {xh_embedded.shape}")

        # Pass through Transformer encoder
        enc_output = xh_embedded
        # print(f"xh_embedded: {xh_embedded}")
        
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, attn_mask)

        # print(f"enc_output: {enc_output}")
        # Reshape back to (bs * n_nodes, d_model * 2)
        enc_output = enc_output.view(bs_n_nodes, self.d_model * 2)
        # print(f"enc_output: {enc_output}")
        # Output heads for noise prediction
        noise_h = self.noise_h_head(enc_output)  # Noise for h (categorical + integer)
        noise_x = self.noise_x_head(enc_output)  # Noise for x (3D coordinates)
        # print(f"noise_x: {noise_x}")
        # print(f"Shape of noise_h: {noise_h.shape}")
        if node_mask is not None:
            # node_mask = node_mask.view(-1, 1)  # Reshape to (bs * n_nodes, 1)
            noise_h = noise_h * node_mask
            noise_x = noise_x * node_mask

        # Combine outputs into a dictionary
        # noise_outputs = {
        #     'noise_h': noise_h,
        #     'noise_x': noise_x
        # }

        # return noise_outputs
        return noise_h, noise_x
    
import torch
import torch.nn as nn
import argparse

def test_transformer():
    # 1. Setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args([])  # Empty list means don't read from command line
    
    # 2. Set model parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_node_nf = 5  # Node feature dimension
    d_model = 64    # Transformer hidden dimension
    num_heads = 8   # Number of attention heads
    num_layers = 4  # Number of transformer layers
    d_ff = 256      # Feed-forward network dimension
    dropout = 0.1   # Dropout rate
    
    # 3. Initialize model
    model = Transformer(
        args=args,
        in_node_nf=in_node_nf,
        device=device,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    
    # 4. Prepare input data
    batch_size = args.batch_size
    n_nodes = 23    # Number of nodes per molecule
    
    # Create random inputs
    h = torch.randn(batch_size * n_nodes, in_node_nf).to(device)  # Node features
    x = torch.randn(batch_size * n_nodes, 3).to(device)           # 3D coordinates
    
    # Create node mask
    node_mask = torch.ones(batch_size, n_nodes, 1).to(device)
    # Assume the last 3 nodes of each molecule are padding
    node_mask[:, -3:, :] = 0
    
    # 5. Forward pass
    # print("Input shapes:")
    # print(f"h shape: {h.shape}")
    # print(f"x shape: {x.shape}")
    # print(f"node_mask shape: {node_mask.shape}")
    
    try:
        noise_h, noise_x = model(h, x, node_mask=node_mask)
        
        print("\nOutput shapes:")
        print(f"noise_h shape: {noise_h.shape}")
        print(f"noise_x shape: {noise_x.shape}")
        
        # 6. Check outputs
        # Verify noise_h dimensions
        assert noise_h.shape == (batch_size * n_nodes, in_node_nf), \
            f"Expected noise_h shape {(batch_size * n_nodes, in_node_nf)}, got {noise_h.shape}"
        
        # Verify noise_x dimensions
        assert noise_x.shape == (batch_size * n_nodes, 3), \
            f"Expected noise_x shape {(batch_size * n_nodes, 3)}, got {noise_x.shape}"
        
        print("\nAll shape checks passed!")
        
        # 7. Check output value ranges
        print("\nOutput statistics:")
        print(f"noise_h mean: {noise_h.mean().item():.3f}, std: {noise_h.std().item():.3f}")
        print(f"noise_x mean: {noise_x.mean().item():.3f}, std: {noise_x.std().item():.3f}")
        
        # 8. Additional checks (optional)
        # Check if outputs for padding nodes are handled correctly
        # You might want to verify that the noise predictions for padding nodes 
        # have appropriate values or are properly masked
        
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
        raise e
    
def test_transformer_padding():
    # 1. Setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args([])
    
    # 2. Set model parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_node_nf = 5
    d_model = 64
    num_heads = 8
    num_layers = 4
    d_ff = 256
    dropout = 0.1
    
    # 3. Initialize model
    model = Transformer(
        args=args,
        in_node_nf=in_node_nf,
        device=device,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    
    # 4. Prepare input data with explicit padding
    batch_size = args.batch_size
    n_nodes = 23
    
    # Create data where some nodes are real and others are padding
    h = torch.randn(batch_size * n_nodes, in_node_nf).to(device)
    x = torch.randn(batch_size * n_nodes, 3).to(device)
    
    # Create node mask: mark last 5 nodes of each molecule as padding
    node_mask = torch.ones(batch_size, n_nodes, 1).to(device)
    node_mask[:, -5:, :] = 0  # Last 5 nodes are padding
    
    # Set padding nodes to zero in input
    h_reshaped = h.view(batch_size, n_nodes, -1)
    x_reshaped = x.view(batch_size, n_nodes, -1)
    
    h_reshaped = h_reshaped * node_mask
    x_reshaped = x_reshaped * node_mask
    
    h = h_reshaped.view(batch_size * n_nodes, -1)
    x = x_reshaped.view(batch_size * n_nodes, -1)
    
    print("\nInput data analysis:")
    print(f"Number of real nodes per molecule: {n_nodes - 5}")
    print(f"Number of padding nodes per molecule: 5")
    
    # 5. Forward pass
    noise_h, noise_x = model(h, x, node_mask=node_mask)
    
    # 6. Analyze outputs
    noise_h_reshaped = noise_h.view(batch_size, n_nodes, -1)
    noise_x_reshaped = noise_x.view(batch_size, n_nodes, -1)
    print(f"\nNoise shape: {noise_h_reshaped.shape}")
    print(f"Output: {noise_h_reshaped[0, ...]}")
    
    # Calculate statistics for real and padding nodes separately
    real_nodes_h = noise_h_reshaped[:, :-5, :]
    padding_nodes_h = noise_h_reshaped[:, -5:, :]
    
    real_nodes_x = noise_x_reshaped[:, :-5, :]
    padding_nodes_x = noise_x_reshaped[:, -5:, :]
    
    print("\nNoise statistics for h:")
    print(f"Real nodes - mean: {real_nodes_h.mean():.3f}, std: {real_nodes_h.std():.3f}")
    print(f"Padding nodes - mean: {padding_nodes_h.mean():.3f}, std: {padding_nodes_h.std():.3f}")
    
    print("\nNoise statistics for x:")
    print(f"Real nodes - mean: {real_nodes_x.mean():.3f}, std: {real_nodes_x.std():.3f}")
    print(f"Padding nodes - mean: {padding_nodes_x.mean():.3f}, std: {padding_nodes_x.std():.3f}")
    
    # Check if padding nodes have near-zero predictions
    padding_threshold = 1e-6
    h_padding_close_to_zero = torch.abs(padding_nodes_h).mean() < padding_threshold
    x_padding_close_to_zero = torch.abs(padding_nodes_x).mean() < padding_threshold
    
    print("\nPadding analysis:")
    print(f"Average absolute value for h padding nodes: {torch.abs(padding_nodes_h).mean():.6f}")
    print(f"Average absolute value for x padding nodes: {torch.abs(padding_nodes_x).mean():.6f}")
    print(f"Padding predictions effectively zero (threshold={padding_threshold}):")
    print(f"h: {h_padding_close_to_zero}")
    print(f"x: {x_padding_close_to_zero}")
import torch
import torch.nn as nn
import argparse
import numpy as np

def test_transformer_permutation_invariance():
    """
    Test if the Transformer model exhibits permutation invariance with respect to node ordering.
    If the model produces the same output for the same data with different node orderings,
    it demonstrates permutation invariance.
    """
    print("Testing Transformer for Permutation Invariance")
    print("-" * 50)
    
    # 1. Setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)  # Using smaller batch size for debugging
    parser.add_argument('--predict_charges', type=bool, default=False)
    parser.add_argument('--include_charges', type=bool, default=False)
    args = parser.parse_args([])
    
    # 2. Set model parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_node_nf = 5      # Node feature dimension
    d_model = 64        # Transformer hidden dimension
    num_heads = 8       # Number of attention heads
    num_layers = 4      # Number of transformer layers
    d_ff = 256          # Feed-forward network dimension
    dropout = 0.1       # Dropout rate
    
    # 3. Initialize model
    model = Transformer(
        args=args,
        in_node_nf=in_node_nf,
        device=device,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    
    # 4. Prepare input data
    batch_size = args.batch_size
    n_nodes = 10        # Number of nodes per molecule
    
    # Create random input data
    torch.manual_seed(42)  # Fix random seed for reproducibility
    h = torch.randn(batch_size * n_nodes, in_node_nf).to(device)  # Node features
    x = torch.randn(batch_size * n_nodes, 3).to(device)           # 3D coordinates
    
    # Create node mask
    node_mask = torch.ones(batch_size * n_nodes, 1).to(device)  # All nodes are valid
    
    # 5. Forward pass with original order
    print("Running forward pass with original node ordering...")
    noise_h_original, noise_x_original = model(h, x, node_mask=node_mask, batch_size=batch_size)
    
    # 6. Generate permutation indices for each molecule
    all_perm_indices = []
    for b in range(batch_size):
        # Generate a random permutation for each molecule
        indices = np.arange(n_nodes)
        np.random.shuffle(indices)
        # Adjust indices to include batch dimension offset
        batch_offset = b * n_nodes
        indices = indices + batch_offset
        all_perm_indices.append(indices)
    
    # Concatenate permutation indices for all batches
    perm_indices = np.concatenate(all_perm_indices)
    
    # 7. Create permuted inputs
    h_permuted = h[perm_indices].clone()
    x_permuted = x[perm_indices].clone()
    node_mask_permuted = node_mask[perm_indices].clone()
    
    # 8. Forward pass with permuted inputs
    print("Running forward pass with permuted node ordering...")
    noise_h_permuted, noise_x_permuted = model(h_permuted, x_permuted, node_mask=node_mask_permuted, batch_size=batch_size)
    
    # 9. Create inverse permutation mapping to restore original order
    inv_perm_indices = np.zeros_like(perm_indices)
    for i, idx in enumerate(perm_indices):
        inv_perm_indices[idx] = i
    
    # 10. Restore permuted outputs to original order
    noise_h_restored = noise_h_permuted[inv_perm_indices].clone()
    noise_x_restored = noise_x_permuted[inv_perm_indices].clone()
    
    # 11. Compare original outputs with restored outputs
    h_diff = torch.abs(noise_h_original - noise_h_restored).mean().item()
    x_diff = torch.abs(noise_x_original - noise_x_restored).mean().item()
    
    print("\nResult Analysis:")
    print(f"Node features (h) output mean difference: {h_diff:.6f}")
    print(f"Spatial coordinates (x) output mean difference: {x_diff:.6f}")
    
    # 12. Set a tolerance threshold to determine if differences are significant
    tolerance = 1e-5
    if h_diff < tolerance and x_diff < tolerance:
        print("\nTest PASSED! ✓")
        print(f"Model exhibits permutation invariance for node ordering (difference < {tolerance}).")
        return True
    else:
        print("\nTest FAILED! ✗")
        print(f"Model does not exhibit permutation invariance for node ordering (difference > {tolerance}).")
        
        # Provide more detailed analysis of differences
        print("\nDetailed Difference Analysis:")
        h_max_diff = torch.abs(noise_h_original - noise_h_restored).max().item()
        x_max_diff = torch.abs(noise_x_original - noise_x_restored).max().item()
        print(f"Node features (h) maximum difference: {h_max_diff:.6f}")
        print(f"Spatial coordinates (x) maximum difference: {x_max_diff:.6f}")
        
        # Check the position of the first different element
        h_diff_mask = torch.abs(noise_h_original - noise_h_restored) > tolerance
        x_diff_mask = torch.abs(noise_x_original - noise_x_restored) > tolerance
        
        if h_diff_mask.any():
            idx = h_diff_mask.nonzero()[0]
            print(f"\nFirst different h output at position: {idx.tolist()}")
            print(f"Original h output value: {noise_h_original[idx[0], idx[1]].item():.6f}")
            print(f"Restored h output value: {noise_h_restored[idx[0], idx[1]].item():.6f}")
        
        if x_diff_mask.any():
            idx = x_diff_mask.nonzero()[0]
            print(f"\nFirst different x output at position: {idx.tolist()}")
            print(f"Original x output value: {noise_x_original[idx[0], idx[1]].item():.6f}")
            print(f"Restored x output value: {noise_x_restored[idx[0], idx[1]].item():.6f}")
        return False

def test_transformer_permutation_group():
    """
    Test if the Transformer model exhibits permutation invariance with respect to molecule ordering.
    This checks if the model can maintain invariance to the overall ordering of molecules
    while preserving the internal structure of each molecule.
    """
    print("\nTesting Transformer for Batch Permutation Invariance")
    print("-" * 50)
    
    # 1. Setup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--predict_charges', type=bool, default=False)
    parser.add_argument('--include_charges', type=bool, default=False)
    args = parser.parse_args([])
    
    # 2. Set model parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_node_nf = 5
    d_model = 64
    num_heads = 8
    num_layers = 4
    d_ff = 256
    dropout = 0.1
    
    # 3. Initialize model
    model = Transformer(
        args=args,
        in_node_nf=in_node_nf,
        device=device,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    )
    
    # 4. Prepare input data
    batch_size = args.batch_size
    n_nodes = 10
    
    torch.manual_seed(42)
    h = torch.randn(batch_size * n_nodes, in_node_nf).to(device)
    x = torch.randn(batch_size * n_nodes, 3).to(device)
    
    # Reshape data to (batch_size, n_nodes, feature_dim) for easier manipulation
    h_batched = h.view(batch_size, n_nodes, -1)
    x_batched = x.view(batch_size, n_nodes, -1)
    
    # Create node mask
    node_mask = torch.ones(batch_size * n_nodes, 1).to(device)
    
    # 5. Forward pass with original batch order
    print("Running forward pass with original batch ordering...")
    noise_h_original, noise_x_original = model(h, x, node_mask=node_mask, batch_size=batch_size)
    
    # 6. Permute the order of entire batches (molecules)
    batch_indices = np.arange(batch_size)
    np.random.shuffle(batch_indices)
    
    # Create permuted inputs
    h_permuted_batches = h_batched[batch_indices].clone()
    x_permuted_batches = x_batched[batch_indices].clone()
    
    # Reshape data back to (batch_size * n_nodes, feature_dim)
    h_permuted = h_permuted_batches.view(batch_size * n_nodes, -1)
    x_permuted = x_permuted_batches.view(batch_size * n_nodes, -1)
    
    # Node mask also needs to be permuted by batch
    node_mask_batched = node_mask.view(batch_size, n_nodes, -1)
    node_mask_permuted_batches = node_mask_batched[batch_indices].clone()
    node_mask_permuted = node_mask_permuted_batches.view(batch_size * n_nodes, -1)
    
    # 7. Forward pass with permuted batches
    print("Running forward pass with permuted batch ordering...")
    noise_h_permuted, noise_x_permuted = model(h_permuted, x_permuted, node_mask=node_mask_permuted, batch_size=batch_size)
    
    # 8. Create inverse permutation mapping to restore original batch order
    inv_batch_indices = np.zeros_like(batch_indices)
    for i, idx in enumerate(batch_indices):
        inv_batch_indices[idx] = i
    
    # Reshape outputs to (batch_size, n_nodes, output_dim)
    noise_h_permuted_batched = noise_h_permuted.view(batch_size, n_nodes, -1)
    noise_x_permuted_batched = noise_x_permuted.view(batch_size, n_nodes, -1)
    
    # Apply inverse permutation
    noise_h_restored_batched = noise_h_permuted_batched[inv_batch_indices].clone()
    noise_x_restored_batched = noise_x_permuted_batched[inv_batch_indices].clone()
    
    # Reshape outputs back to (batch_size * n_nodes, output_dim)
    noise_h_restored = noise_h_restored_batched.view(batch_size * n_nodes, -1)
    noise_x_restored = noise_x_restored_batched.view(batch_size * n_nodes, -1)
    
    # 9. Compare original outputs with restored outputs
    h_diff = torch.abs(noise_h_original - noise_h_restored).mean().item()
    x_diff = torch.abs(noise_x_original - noise_x_restored).mean().item()
    
    print("\nResult Analysis:")
    print(f"After batch permutation, node features (h) output mean difference: {h_diff:.6f}")
    print(f"After batch permutation, spatial coordinates (x) output mean difference: {x_diff:.6f}")
    
    # 10. Set a tolerance threshold
    tolerance = 1e-5
    if h_diff < tolerance and x_diff < tolerance:
        print("\nTest PASSED! ✓")
        print(f"Model exhibits permutation invariance for batch ordering (difference < {tolerance}).")
        return True
    else:
        print("\nTest FAILED! ✗")
        print(f"Model does not exhibit permutation invariance for batch ordering (difference > {tolerance}).")
        return False

if __name__ == "__main__":
    # Run node permutation invariance test
    node_permutation_result = test_transformer_permutation_invariance()
    
    # Run molecule group permutation test
    group_permutation_result = test_transformer_permutation_group()
    
    print("\nSummary:")
    if node_permutation_result and group_permutation_result:
        print("All tests passed. Model exhibits complete permutation invariance.")
    elif node_permutation_result:
        print("Model exhibits permutation invariance for node ordering, but not for batch ordering.")
    elif group_permutation_result:
        print("Model exhibits permutation invariance for batch ordering, but not for node ordering.")
    else:
        print("Model does not exhibit permutation invariance for any type of ordering. Architectural changes may be needed.")