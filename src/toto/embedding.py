import torch
import torch.nn as nn



## Patch Embedding ##
## --------------- ##

class PatchEmbedding(nn.Module):
    """
    A simple implementation of the PatchEmbedding mechanism.
    Patchifies a multivariate time series and projects the patches.
    """

    def __init__(self, patch_size: int, stride: int, embed_dim: int):
        """
        Initializes the PatchEmbedding module.

        Args:
            patch_size: The size of each patch.
            stride: The step size between patches, i.e. if the first patch 
            starts at index 0, the next will start at index 0 + stride.
            embed_dim: The dimension to project the patches into.
        """
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride

        self.projection = nn.Linear(self.patch_size, self.embed_dim)

    def _patchify(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Splits the input tensor into patches. 

        Args:
            x: The input time series tensor.

        Returns:
            The patched tensor with dimension (batch, variate, num_patches, patch_size). 
        """
        return x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
    
    def forward(
        self,
        x: torch.Tensor,
        id_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Performs the forward pass of the patch embedding.

        Args:
            x: The input time series tensor with shape (batch, variate, time_steps).
            id_mask: The identity mask tensor.

        Returns:
            A tuple containing the projected embeddings (batch, variate, num_patch, embed_dim)
            and the patched identity mask.
        """
        # Ensure the time steps are divisible by the patch size.
        assert (
            x.shape[-1] % self.patch_size == 0
        ), f"Series length ({x.shape[-1]}) must be divisible by patch size ({self.patch_size})"

        x_patched       = self._patchify(x)
        id_mask_patched = self._patchify(id_mask)

        # Ensure that patches don't span multiple identities.
        assert torch.eq(
            id_mask_patched.min(-1).values, id_mask_patched.max(-1).values
        ).all(), "Patches cannot span multiple datasets"

        # Project the patches and return the projected patches along with the patched mask.
        return (
            self.projection(x_patched),
            id_mask_patched.min(-1).values,
        )
     
## Rotary Embedding ## 
## ---------------- ##


class RotaryEmbedding(nn.Module):
    """
    Implements Rotary Position Embeddings (RoPE) for queries and keys.

    RoPE works by rotating the embedding vectors of each token based on its
    position in the sequence. This implicitly encodes relative positional
    information, making it more robust and effective than traditional
    positional encodings.
    """
    def __init__(self, head_dim: int, max_seq_len: int = 512):
        """
        Initializes the RotaryEmbedding module.

        Args:
            head_dim (int): The dimension of a single attention head.
            max_seq_len (int): The maximum sequence length to pre-compute
                               the rotation angles for.
        """
        super().__init__()
        self.head_dim = head_dim
        
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_sin", None)
        self.register_buffer("cached_cos", None)
        
        self.max_seq_len = max_seq_len
        self._set_cached_rotary_angles(max_seq_len)

    def _set_cached_rotary_angles(self, seq_len: int):
        """
        Pre-computes and caches the sine and cosine values for the rotation.

        This is done to avoid re-computing these values on every forward pass,
        which improves efficiency.
        """
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        
        self.cached_sin = emb.sin()[None, None, :, :]  # Reshape to (1, 1, seq_len, head_dim)
        self.cached_cos = emb.cos()[None, None, :, :]  # Reshape to (1, 1, seq_len, head_dim)

    def _rotate_tensor(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """
        Applies the rotation to a tensor.

        Args:
            x (torch.Tensor): The input tensor to rotate (e.g., queries or keys).
                              Shape: (batch_size, num_heads, seq_len, head_dim).
            sin (torch.Tensor): Pre-computed sine values.
            cos (torch.Tensor): Pre-computed cosine values.

        Returns:
            torch.Tensor: The rotated tensor.
        """
        # Split the tensor into two halves along the last dimension.
        # This is a common RoPE implementation trick to apply 2D rotations
        # to higher-dimensional vectors.
        x_even, x_odd = x.chunk(2, dim=-1)
        
        # We need to make sure the sin and cos tensors are also split correctly.
        # Since they are of shape (1, 1, seq_len, head_dim), we split them on dim=-1.
        cos_even, cos_odd = cos.chunk(2, dim=-1)
        sin_even, sin_odd = sin.chunk(2, dim=-1)
        
        # The rotation logic: new_x = x_even * cos - x_odd * sin
        #                    new_y = x_even * sin + x_odd * cos
        x_even_rotated = x_even * cos_even - x_odd * sin_even
        x_odd_rotated = x_even * sin_odd + x_odd * cos_odd
        
        # Concatenate the rotated halves back together.
        return torch.cat((x_even_rotated, x_odd_rotated), dim=-1)

    def rotate_queries_and_keys(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Rotates the query and key tensors using Rotary Embeddings.

        Args:
            q (torch.Tensor): Query tensor.
                              Shape: (batch_size, num_heads, seq_len, head_dim).
            k (torch.Tensor): Key tensor.
                              Shape: (batch_size, num_heads, seq_len, head_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The rotated queries and keys.
        """
        # If the current sequence length is longer than the pre-computed
        # length, we recompute the angles.
        seq_len = q.shape[-2]
        if seq_len > self.max_seq_len:
            self.max_seq_len = seq_len
            self._set_cached_rotary_angles(seq_len)

        # Slice the cached angles to match the current sequence length.
        sin = self.cached_sin[..., :seq_len, :]
        cos = self.cached_cos[..., :seq_len, :]

        # Rotate the queries and keys.
        q_rotated = self._rotate_tensor(q, sin, cos)
        k_rotated = self._rotate_tensor(k, sin, cos)
        
        return q_rotated, k_rotated