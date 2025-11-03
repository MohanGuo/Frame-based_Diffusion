import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Transformer(nn.Module):
    def __init__(self, args, in_node_nf, device, d_model, num_heads, num_layers, d_ff, dropout, edge_dim):
        super(Transformer, self).__init__()
        self.device = device
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)

        # Embedding layers for h and x
        self.h_embedding = nn.Linear(in_node_nf, d_model)
        self.x_embedding = nn.Linear(3, d_model)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Standard TransformerEncoderLayer
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model * 2,  # Concatenated h and x embeddings
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="relu",
                batch_first=False  # Keep (seq_len, batch_size, d_model) format
            )
            for _ in range(num_layers)
        ])

        # Edge feature processing
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),  # First LayerNorm
            nn.ReLU(),
            nn.LayerNorm(d_model * 2),  # Second LayerNorm (optional)
            nn.Linear(d_model * 2, num_heads)  # Project to attention heads
        )

        # Output heads for noise prediction
        self.noise_h_head = nn.Linear(d_model, in_node_nf)
        self.noise_x_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3)
        )

        self.to(device)

    def generate_mask(self, h, x, node_mask=None, batch_size=0):
        if node_mask is None:
            return None

        total_nodes = node_mask.shape[0]
        n_nodes = total_nodes // batch_size

        node_mask_i = node_mask.view(batch_size, n_nodes, 1)
        node_mask_j = node_mask.view(batch_size, 1, n_nodes)
        attention_mask = node_mask_i * node_mask_j

        # Expand mask for multi-head attention
        attention_mask = attention_mask.unsqueeze(1)  # Shape: (batch_size, 1, n_nodes, n_nodes)
        attention_mask = attention_mask.expand(-1, self.num_heads, -1, -1)  # Shape: (batch_size, num_heads, n_nodes, n_nodes)
        attention_mask = attention_mask.reshape(batch_size * self.num_heads, n_nodes, n_nodes)  # Shape: (batch_size * num_heads, n_nodes, n_nodes)

        return attention_mask

    def generate_edge_bias(self, edge_features, node_mask=None, edge_mask=None, batch_size=0, n_nodes=0):
        """Generate attention bias from edge features"""
        if edge_features is None:
            return 0.0  # Return a scalar that won't affect addition
            
        # Process edge features - shape: [batch_size, n_nodes, n_nodes, edge_dim]
        edge_bias = self.edge_proj(edge_features)  # [batch_size, n_nodes, n_nodes, num_heads]
        
        # Reshape to match attention mask format
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # [batch_size, num_heads, n_nodes, n_nodes]
        
        # Apply edge mask if provided
        if edge_mask is not None:
            edge_mask = edge_mask.unsqueeze(1)  # [batch_size, 1, n_nodes, n_nodes]
            edge_bias = edge_bias * edge_mask
            
        # Reshape to match attention mask shape used in TransformerEncoderLayer
        edge_bias = edge_bias.reshape(batch_size * self.num_heads, n_nodes, n_nodes)
        
        return edge_bias

    def forward(self, t, h, x, edge_features, node_mask=None, edge_mask=None, batch_size=0, debug=False):
        bs_n_nodes = x.size(0)
        n_nodes = bs_n_nodes // batch_size

        t_embedded = self.t_embedding(t)
        
        # Embed h and x
        h_embedded = self.dropout(self.h_embedding(h))
        x_embedded = self.dropout(self.x_embedding(x))

        # Concatenate h and x embeddings
        xh_embedded = torch.cat([h_embedded, x_embedded], dim=-1)  # (bs * n_nodes, d_model * 2)

        # Reshape for Transformer input: (seq_len, batch_size, d_model * 2)
        xh_embedded = xh_embedded.view(batch_size, bs_n_nodes // batch_size, self.d_model * 2)
        xh_embedded = xh_embedded.transpose(0, 1)  # (seq_len, batch_size, d_model * 2)

        # Generate standard mask
        attn_mask = self.generate_mask(h, x, node_mask, batch_size)
        
        # Generate edge bias (if edge_features is provided)
        # edge_features = None
        if edge_features is not None:
            edge_bias = self.generate_edge_bias(edge_features, node_mask, edge_mask, batch_size, n_nodes)
            
            # Convert binary mask to additive mask if needed
            if attn_mask is not None:
                # Convert from boolean mask (1=keep, 0=mask) to additive mask (0=keep, -inf=mask)
                float_mask = attn_mask.float()
                # float_mask = float_mask.masked_fill(float_mask == 0, float('-inf'))
                float_mask = float_mask.masked_fill(float_mask == 0, -1e9)
                float_mask = float_mask.masked_fill(float_mask > 0, 0.0)
                
                # Add edge bias (will only affect non-masked positions)
                attn_mask = float_mask + edge_bias
                # print(f"float_mask: {float_mask}")
            else:
                attn_mask = edge_bias
            
            # print(f"edge_bias: {edge_bias}")
            # print(f"attn_mask: {attn_mask}")
        
        # Pass through TransformerEncoderLayer
        enc_output = xh_embedded

        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask=attn_mask)

        # Reshape back to (bs * n_nodes, d_model * 2)
        enc_output = enc_output.transpose(0, 1).contiguous().view(bs_n_nodes, self.d_model * 2)

        # Output heads for noise prediction
        noise_x = self.noise_x_head(enc_output[..., :self.d_model])
        noise_h = self.noise_h_head(enc_output[..., self.d_model:])

        if node_mask is not None:
            noise_h = noise_h * node_mask
            noise_x = noise_x * node_mask
            
        return noise_h, noise_x

        # # Generate masks (if needed)
        # attn_mask = self.generate_mask(h, x, node_mask, batch_size)

        # # Process edge features
        # edge_scores = self.edge_proj(edge_features)  # (batch_size, num_heads, seq_len, seq_len)
        # edge_scores = edge_scores.permute(0, 2, 3, 1)  # (batch_size, seq_len, seq_len, num_heads)

        # # Pass through TransformerEncoderLayer
        # enc_output = xh_embedded
        # for layer in self.encoder_layers:
        #     enc_output = layer(enc_output, src_mask=attn_mask)

        # # Reshape back to (bs * n_nodes, d_model * 2)
        # enc_output = enc_output.transpose(0, 1).contiguous().view(bs_n_nodes, self.d_model * 2)

        # # Output heads for noise prediction
        # noise_x = self.noise_x_head(enc_output[..., :self.d_model])
        # noise_h = self.noise_h_head(enc_output[..., self.d_model:])

        # if node_mask is not None:
        #     noise_h = noise_h * node_mask
        #     noise_x = noise_x * node_mask

        # return noise_h, noise_x
        