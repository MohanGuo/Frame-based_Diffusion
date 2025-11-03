import math
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter
from torch_geometric.nn import global_mean_pool

class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution with mean and logvar parameters.

    Adapted from: https://github.com/CompVis/latent-diffusion, with modifications for our tensors,
    which are of shape (N, d) instead of (B, H, W, d) for 2D images.
    """

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)  # split along channel dim
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=1
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=1,
                )

    def mode(self):
        return self.mean

    def __repr__(self):
        return f"DiagonalGaussianDistribution(mean={self.mean}, logvar={self.logvar})"



def get_index_embedding(indices, emb_dim, max_len=2048):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., num_tokens] of type integer
        emb_dim: dimension of the embeddings to create
        max_len: maximum length

    Returns:
        positional embedding of shape [..., num_tokens, emb_dim]
    """
    K = torch.arange(emb_dim // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / emb_dim))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class TransformerEncoder(nn.Module):
    """Transformer encoder as part of standard Transformer-based VAEs.
    
    Args:
        max_num_elements: Maximum number of elements in the dataset
        d_model: Dimension of the model
        nhead: Number of attention heads
        dim_feedforward: Dimension of the feedforward network
        activation: Activation function to use
        dropout: Dropout rate
        norm_first: Whether to use pre-normalization in Transformer blocks
        bias: Whether to use bias
        num_layers: Number of layers
    """

    def __init__(
        self,
        # max_num_elements=100,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_first: bool = True,
        bias: bool = True,
        num_layers: int = 8,
        in_node_nf: int = 5
    ):
        super().__init__()

        # self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers

        # self.atom_type_embedder = nn.Embedding(max_num_elements, d_model)
        self.h_embedding = nn.Linear(in_node_nf, d_model)
        self.x_embedding = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        activation = {
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
        }[activation]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Data object with the following attributes:
                h (torch.Tensor): features of atoms in the batch. [bs * n_nodes, egnn_feature_dim]
                x (torch.Tensor): Cartesian coordinates of atoms in the batch
                node_mask
                edge_mask
                n_nodes (torch.Tensor): Number of atoms in the batch
                batch (torch.Tensor): Batch index for each atom
        """
        h = self.h_embedding(batch.h)  # (n, d)
        x= self.x_embedding(batch.x)

        # Positional embedding
        # pos_emb = get_index_embedding(batch.token_idx, self.d_model)

        # Convert from PyG batch to dense batch with padding
        # x, token_mask = to_dense_batch(x, batch.batch)

        # Transformer forward pass
        # print(f"x shape: {x.shape}")
        # print(f"node_mask shape: {batch.node_mask.shape}")
        batch_size = batch.node_mask.size(0)  # 32
        seq_length = batch.node_mask.size(1)  # 24
        
        x = x.view(batch_size, seq_length, self.d_model)
        x = self.transformer.forward(x, src_key_padding_mask=(~batch.node_mask))
        # print(f"x output shape: {x.shape}")
        x = x.view(batch_size * seq_length, -1)
        # x = x[token_mask]

        return {
            "x": x,
            "num_atoms": batch.num_atoms,
            "batch": batch.batch,
            # "token_idx": batch.token_idx,
        }

class TransformerDecoder(nn.Module):
    """Transformer decoder as part of pure Transformer-based VAEs.
    
    See src/models/encoders/transformer.py for documentation.
    """

    def __init__(
        self,
        # max_num_elements=100,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        activation: str = "gelu",
        dropout: float = 0.0,
        norm_first: bool = True,
        bias: bool = True,
        num_layers: int = 8,
        in_node_nf: int = 5
    ):
        super().__init__()

        # self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers

        activation = {
            "gelu": nn.GELU(approximate="tanh"),
            "relu": nn.ReLU(),
        }[activation]
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                activation=activation,
                dropout=dropout,
                batch_first=True,
                norm_first=norm_first,
                bias=bias,
            ),
            norm=nn.LayerNorm(d_model),
            num_layers=num_layers,
        )

        # self.atom_types_head = nn.Linear(d_model, max_num_elements, bias=True)
        # self.frac_coords_head = nn.Linear(d_model, 3, bias=False)
        # self.lattice_head = nn.Linear(d_model, 6, bias=False)
        self.h_head = nn.Linear(d_model, in_node_nf)
        self.x_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 3)
        )

    def forward(self, encoded_batch) -> Dict[str, torch.Tensor]:
        """
        Args:
            encoded_batch: Dict with the following attributes:
                x (torch.Tensor): Encoded batch of atomic environments
                num_atoms (torch.Tensor): Number of atoms in each molecular environment
                batch (torch.Tensor): Batch index for each atom
                # token_idx (torch.Tensor): Token index for each atom
        """

        x = encoded_batch["x"]
        h = encoded_batch["h"]

        # Positional embedding
        # x += get_index_embedding(encoded_batch["token_idx"], self.d_model)

        # Convert from PyG batch to dense batch with padding
        # x, token_mask = to_dense_batch(x, encoded_batch["batch"])

        # Transformer forward pass
        node_mask = encoded_batch["node_mask"]
        batch_size = node_mask.size(0)  # 32
        seq_length = node_mask.size(1)  # 24
        
        x = x.view(batch_size, seq_length, -1)
        x = self.transformer.forward(x, src_key_padding_mask=(~node_mask))
        # print(f"x output shape: {x.shape}")
        x = x.view(batch_size * seq_length, -1)
        # x = self.transformer.forward(x, src_key_padding_mask=(~encoded_batch.node_mask))
        # x = x[token_mask]

        # Atomic type prediction head
        h_out = self.h_head(h)

        # Fractional coordinates prediction head
        x_out = self.x_head(x)


        return {
            "h": h_out,
            "x": x_out,
        }


    
# class VAE_Point(nn.Module):

#     def __init__(self, latent_dim=8):
#         super().__init__()
#         self.encoder = TransformerEncoder()
#         self.decoder = TransformerDecoder()

#         # quantization layers (following naming convention from Latent Diffusion)
#         self.quant_conv = torch.nn.Linear(self.encoder.d_model, 2 * latent_dim, bias=False)
#         self.post_quant_conv = torch.nn.Linear(latent_dim, self.decoder.d_model, bias=False)

#     def encode(self, batch):
#         encoded_batch = self.encoder(batch)
#         encoded_batch["moments"] = self.quant_conv(encoded_batch["x"])
#         encoded_batch["posterior"] = DiagonalGaussianDistribution(encoded_batch["moments"])
#         return encoded_batch
    
#     def decode(self, encoded_batch):
#         encoded_batch["x"] = self.post_quant_conv(encoded_batch["x"])
#         out = self.decoder(encoded_batch)
#         return out
    
#     def forward(self, batch):
#         # Encode [B*Num_nodes, C]
#         encoded_batch = self.encode(batch)
#         encoded_batch["x"] = encoded_batch["posterior"].sample()
#         out = self.decode(encoded_batch)
#         return out, encoded_batch

class VAE_Point(nn.Module):
    def __init__(self, latent_dim=8, in_node_nf=5):
        super().__init__()
        #in_node_nf: dims of h
        self.encoder = TransformerEncoder(in_node_nf=in_node_nf)
        self.decoder = TransformerDecoder(in_node_nf=in_node_nf)
        self.latent_dim = latent_dim
        
        d_model = self.encoder.d_model
        
        self.x_quant_conv = torch.nn.Linear(d_model, 2 * latent_dim, bias=False)
        self.x_post_quant_conv = torch.nn.Linear(latent_dim, d_model, bias=False)
        
        self.h_quant_conv = torch.nn.Linear(d_model, 2 * latent_dim, bias=False)
        self.h_post_quant_conv = torch.nn.Linear(latent_dim, d_model, bias=False)

    def encode(self, batch):
        encoded_batch = self.encoder(batch)
        
        x_moments = self.x_quant_conv(encoded_batch["x"])
        x_posterior = DiagonalGaussianDistribution(x_moments)
        
        h_encoded = self.encoder.h_embedding(batch.h)
        h_moments = self.h_quant_conv(h_encoded)
        h_posterior = DiagonalGaussianDistribution(h_moments)
        
        encoded_batch["x_moments"] = x_moments
        encoded_batch["x_posterior"] = x_posterior
        encoded_batch["h_moments"] = h_moments
        encoded_batch["h_posterior"] = h_posterior
        encoded_batch["node_mask"] = batch.node_mask
        
        return encoded_batch
    
    def decode(self, encoded_batch):
        x_z = encoded_batch["x_z"]
        x_decoded = self.x_post_quant_conv(x_z)
        
        h_z = encoded_batch["h_z"]
        h_decoded = self.h_post_quant_conv(h_z)
        
        decoder_input = {
            "x": x_decoded,
            "h": h_decoded,
            "node_mask": encoded_batch["node_mask"],
            "batch": encoded_batch.get("batch", None),
            "num_atoms": encoded_batch.get("num_atoms", None)
        }
        
        out = self.decoder(decoder_input)
        return out
    
    def forward(self, batch, use_mean=False):
        encoded_batch = self.encode(batch)
        
        if use_mean:
            encoded_batch["x_z"] = encoded_batch["x_posterior"].mode()
            encoded_batch["h_z"] = encoded_batch["h_posterior"].mode()
        else:
            encoded_batch["x_z"] = encoded_batch["x_posterior"].sample()
            encoded_batch["h_z"] = encoded_batch["h_posterior"].sample()
        
        out = self.decode(encoded_batch)
        
        return out, encoded_batch