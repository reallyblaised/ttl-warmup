# Imports for the attention module
import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum

from toto.embedding import RotaryEmbedding

## Useful Enum ##
## ----------- ##

class AttentionAxis(Enum):
    """
    Enum to give the different attention heads
    """
    TIME = 1
    SPACE = 2

## Mask Generator ##
## -------------- ##

def make_batched_block_mask(t: torch.Tensor) -> torch.Tensor:
    """
    Creates a batched block mask from a tensor of IDs.
    The mask is True where IDs match, False otherwise.
    """
    unsqueezed = t.unsqueeze(-1)
    return unsqueezed == unsqueezed.transpose(-1, -2)


## The core multiheaded attention mask ##
## ----------------------------------- ## 

class BaseMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        rotary_emb: RotaryEmbedding, # Allow for other forms of positional embedding. 
        use_memory_efficient_attention: bool,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.rotary_emb = rotary_emb
        self.use_memory_efficient_attention = use_memory_efficient_attention # TODO: Not currently implimented, 

        self.wQKV = nn.Linear(embed_dim, embed_dim * 3) # In order to not split the tensors, just keep the tensors as one object
        self.wO = nn.Linear(embed_dim, embed_dim) # Finally linear projection. 

        self.attention_axis = None # Is attention along the special or temporal axis. 

    def _rearrange_inputs(self, inputs: torch.Tensor):
        """
        Rearranges the input tensor based on the attention axis.

        The input will be 4 dim, for the attention mechanism to work it will
        need to be 3 dim. To do this TOTO compresses the batch dimension with 
        variate variable for time attn and seq for space. 
        """
        if self.attention_axis == AttentionAxis.TIME:
            return inputs.reshape(-1, inputs.shape[2], inputs.shape[3])
        elif self.attention_axis == AttentionAxis.SPACE:
            return inputs.permute(0, 2, 1, 3).reshape(-1, inputs.shape[1], inputs.shape[3])

    def _split_heads(self, x):
        """
        Splits the projected QKV tensor into heads and reshapes it.
        """
        batch_size, seq_len, _ = x.shape
        return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _combine_heads(self, x):
        """
        Combines the heads back into a single tensor for the output projection.
        """
        batch_size, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        The most important method.

        The input is expected as a 4 dim tensor with axis (B,V,S,E). 
        Depeding on the attention axis, this will be collapsed via the 
        _rarrange_inputs function. 
        """
        
        batch_size, variate, seq_len, embed_dim = inputs.shape
        dropout = self.dropout if self.training else 0.0

        # Rearrange input tensor based on the attention axis
        rearranged_inputs = self._rearrange_inputs(inputs)

        # Project input to Q, K, and V
        qkv = self.wQKV(rearranged_inputs)
        
        qkv = qkv.view(rearranged_inputs.shape[0], rearranged_inputs.shape[1], 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        # Transpose to (batch, num_heads, seq_len, head_dim) for F.scaled_dot_product_attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rotary_emb and self.attention_axis == AttentionAxis.TIME: 
            q, k = self.rotary_emb.rotate_queries_and_keys(q, k)
        
        # Perform scaled dot-product attention
        if self.attention_axis == AttentionAxis.TIME:
            # Causal attention for time-wise
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout,
                is_causal=True,
            )
        else:
            # Bidirectional attention for space-wise
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=dropout 
            )

        # Reshape the output and apply final linear projection
        output = self._combine_heads(output)

        # Reshape the output back to the original shape
        if self.attention_axis == AttentionAxis.TIME:
            output = output.reshape(batch_size, variate, seq_len, embed_dim)
        elif self.attention_axis == AttentionAxis.SPACE:
            output = output.reshape(batch_size, seq_len, variate, embed_dim).permute(0, 2, 1, 3)

        return self.wO(output)

## Time wise attention ##
## ------------------- ##

class TimeWiseMultiheadAttention(BaseMultiheadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_axis = AttentionAxis.TIME

## Space wise attention ##
## -------------------- ##

class SpaceWiseMultiheadAttention(BaseMultiheadAttention):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attention_axis = AttentionAxis.SPACE


def make_batched_block_mask(t: torch.Tensor) -> torch.Tensor:
    """
    Creates a batched block mask from a 1D tensor of IDs.
    The mask is True where IDs match, False otherwise.
    
    This function is used to create a mask that prevents attention between
    different identities (e.g., different cities in the same batch).
    """
    unsqueezed = t.unsqueeze(-1)
    return unsqueezed == unsqueezed.transpose(-1, -2)


def make_space_mask(id_mask: torch.Tensor) -> torch.Tensor:
    """
    Creates a space mask for attention, ensuring attention is restricted
    to data points from the same identity within the batch.

    Args:
        id_mask (torch.Tensor): Identity mask with shape (B, V, S).

    Returns:
        torch.Tensor: Space mask with shape (B * S, 1, V, V).
    """
    batch, variate, seq_len = id_mask.shape

    # Get block mask per batch (B, V, V)
    block_mask = make_batched_block_mask(id_mask[:, :, 0])

    # Expand over sequence and reshape directly
    final_mask = block_mask.unsqueeze(1).expand(batch, seq_len, variate, variate)
    final_mask = final_mask.reshape(batch * seq_len, 1, variate, variate)

    return final_mask

