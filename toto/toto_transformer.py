"""
This script contains Toto transformer layer and decoder block. 
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from toto.attention import AttentionAxis, TimeWiseMultiheadAttention, SpaceWiseMultiheadAttention, make_space_mask
from toto.normalise import RMSNorm 
from toto.embedding import RotaryEmbedding 



## SwigLu ##
## ------ ##

class SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, bias: bool = True):
        super().__init__()
        
        self.linear_gate = nn.Linear(in_features, hidden_features, bias=bias)
        self.linear_data = nn.Linear(in_features, hidden_features, bias=bias)
    
        self.linear_out = nn.Linear(hidden_features, in_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        gate = F.silu(self.linear_gate(x))
        data = self.linear_data(x)
        gated_output = gate * data
    
        return self.linear_out(gated_output)

## Individual Transformer Layer ## 
## ---------------------------- ## 

class TotoTransformerLayer(nn.Module):
    """
    A transformer block that applies multi-head attention followed by a feed-forward network. 
    
    This module uses pre-normalization, where normalization is applied before each sublayer.
    
    Note: The use of memory-efficient attention is currently disabled for simplicity and will be
    implemented in a future update.

    Note: This class currently has a RMS pre-norm hard coded, along with skip connections. Please 
    update this in the future to be more flexiable. 

    Args:
        embed_dim (int): The dimensionality of the input and output embeddings.
        num_heads (int): The number of attention heads.
        mlp_hidden_dim (int): The hidden dimensionality of the feed-forward network.
        dropout (float): The dropout probability.
        rotary_emb (RotaryEmbedding): The rotary positional embedding module.
        attention_axis (AttentionAxis): The axis along which attention is applied (TIME or SPACE).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float,
        rotary_emb: RotaryEmbedding,
        attention_axis: AttentionAxis,
    ):
        super().__init__()

        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

        if attention_axis == AttentionAxis.TIME:
            self.attention = TimeWiseMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                rotary_emb=rotary_emb,
                use_memory_efficient_attention=False,
            )
        elif attention_axis == AttentionAxis.SPACE:
            self.attention = SpaceWiseMultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                rotary_emb=None, 
                use_memory_efficient_attention=False,
            )
        else:
            raise ValueError("Invalid attention axis provided.")

        self.mlp = nn.Sequential(
            SwiGLU(in_features=embed_dim, hidden_features=mlp_hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, inputs: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs a forward pass through the transformer layer.

        Args:
            inputs (torch.Tensor): The input tensor.
            attention_mask (torch.Tensor, optional): The mask to apply to attention weights.
                                                   Defaults to None.

        Returns:
            torch.Tensor: The output tensor after passing through the layer.
        """
        # First sublayer: Multi-head Attention with residual connection
        attention_pre_norm = self.norm1(inputs)
        print('here we are', attention_pre_norm.shape)
        hidden_state = inputs + self.attention(attention_pre_norm, attention_mask=attention_mask)

        # Second sublayer: Feed-Forward Network with residual connection
        mlp_pre_norm = self.norm2(hidden_state)
        output = hidden_state + self.mlp(mlp_pre_norm)

        return output
    
## Decoder Block ##
## ------------- ## 
    
class TotoTransformer(nn.Module):
    """
    A Toto decoder block consisting of a sequence of time-wise and space-wise transformer layers.
    
    This block processes an input sequence first with time-wise attention layers (which typically 
    enforce a causal mask for autoregressive tasks) and then with space-wise attention layers.

    Args:
        configs (dict): A dictionary containing model configurations.
                        Expected keys: 'embed_dim', 'num_heads', 'mlp_hidden_dim',
                                       'dropout', 'num_time_layers', 'num_space_layers'.
    """
    def __init__(
            self, 
            embed_dim: int, 
            num_heads: int, 
            mlp_hidden_dim: int, 
            dropout: float, 
            num_time_layers: int, 
            num_space_layers: int
    ):
        super().__init__()

        # Time-wise attention uses rotary embeddings
        rotary_emb = RotaryEmbedding(embed_dim // num_heads)

        self.time_transformers = nn.ModuleList([
            TotoTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                rotary_emb=rotary_emb,
                attention_axis=AttentionAxis.TIME,
            ) for _ in range(num_time_layers)
        ])

        # Space-wise attention does not use rotary embeddings
        self.space_transformers = nn.ModuleList([
            TotoTransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_hidden_dim=mlp_hidden_dim,
                dropout=dropout,
                rotary_emb=None, 
                attention_axis=AttentionAxis.SPACE,
            ) for _ in range(num_space_layers)
        ])

    def forward(self, inputs: torch.Tensor, id_mask: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the decoder block.

        Args:
            inputs (torch.Tensor): The input tensor. Expected shape: (B, V, S, E).
            id_mask (torch.Tensor): The identity mask for space-wise attention.

        Returns:
            torch.Tensor: The output tensor after passing through the time and space layers.
        """
        x = self._time_attend(inputs=inputs)


        space_mask = make_space_mask(id_mask)
        x = self._space_attend(x, attention_mask=space_mask)

        return x
    
    def _time_attend(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Passes inputs through the time-wise attention blocks.
        
        Args:
            inputs (torch.Tensor): The input tensor.
            
        Returns:
            torch.Tensor: The output tensor after time-wise attention.
        """
        for layer in self.time_transformers:
            print(len(self.time_transformers))
            inputs = layer(inputs) # Causal masking is handled implicitly within the TimeWiseMultiheadAttention module.
        
        return inputs 
    
    def _space_attend(self, inputs: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor: 
        """
        Passes inputs through the space-wise attention blocks.
        
        Args:
            inputs (torch.Tensor): The input tensor.
            attention_mask (torch.Tensor): The mask for space-wise attention.
            
        Returns:
            torch.Tensor: The output tensor after space-wise attention.
        """
        for layer in self.space_transformers:
            inputs = layer(inputs, attention_mask)
        
        return inputs