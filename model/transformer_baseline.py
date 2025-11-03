import torch
import torch.nn as nn
import math

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

        # Standard TransformerEncoderLayer
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model * 2,  # Concatenated h and x embeddings
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="relu",
                batch_first=True
            )
            for _ in range(num_layers)
        ])


        # Output heads for noise prediction
        self.noise_h_head = nn.Linear(d_model * 2, in_node_nf)
        self.noise_x_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3)
        )

        self.to(device)

    def generate_mask(self, h, node_mask=None, batch_size=0):
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

    def forward(self, h, x, node_mask=None, edge_mask=None, batch_size=0, debug=False):
        bs_n_nodes = h.size(0)
        n_nodes = bs_n_nodes // batch_size

        # Embed h and x
        h_embedded = self.dropout(self.h_embedding(h))
        x_embedded = self.dropout(self.x_embedding(x))

        # Concatenate h and x embeddings
        xh_embedded = torch.cat([h_embedded, x_embedded], dim=-1)  # (bs * n_nodes, d_model * 2)
        # print(f"xh_embedded: {xh_embedded.shape}")
        # print(f"h: {h.shape}")
        xh_embedded = xh_embedded.view(batch_size, n_nodes, self.d_model * 2)


        # Reshape for Transformer input: (seq_len, batch_size, d_model * 2)
        # h_embedded = h_embedded.view(batch_size, bs_n_nodes // batch_size, self.d_model * 2)
        # h_embedded = h_embedded.transpose(0, 1)  # (seq_len, batch_size, d_model * 2)

        # Generate standard mask
        # attn_mask = self.generate_mask(h, node_mask, batch_size)
        key_padding_mask = None
        if node_mask is not None:
            #(batch_size, seq_len): padding
            key_padding_mask = ~(node_mask.bool().view(batch_size, n_nodes))

        
        # Pass through TransformerEncoderLayer
        enc_output = xh_embedded

        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_key_padding_mask=key_padding_mask)

        # Reshape back to (bs * n_nodes, d_model * 2)
        enc_output = enc_output.contiguous().view(bs_n_nodes, self.d_model * 2)

        # Output heads for noise prediction
        noise_x = self.noise_x_head(enc_output)
        noise_h = self.noise_h_head(enc_output)

        if node_mask is not None:
            noise_h = noise_h * node_mask
            noise_x = noise_x * node_mask
            
        return noise_h, noise_x