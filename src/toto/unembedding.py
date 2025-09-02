import torch
import torch.nn as nn 



class Unembed(nn.Module):
    """
    A simple unembedding layer for a transformer model.

    This module takes the transformer's output, applies a linear projection,
    and then rearranges the tensor to flatten the patched sequence
    back into a continuous time series representation.

    Parameters
    ----------
    embed_dim : int
        The dimension of the model's latent space (the transformer output).
    patch_size : int
        The size of the patch used in the initial embedding step.
    """
    def __init__(self, embed_dim: int, patch_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        
        self.unembed_linear = nn.Linear(self.embed_dim, self.patch_size)

    def forward(
        self, inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the unembedding and flattening of the tensor.

        Args:
            inputs: The input tensor from the transformer's final layer,
                    with shape (batch, variate, seq_len, embed_dim).

        Returns:
            The unembedded and flattened tensor with shape
            (batch, variate, time_steps, embed_dim), where time_steps
            is equal to seq_len * patch_size.
        """

        unembedded_output = self.unembed_linear(inputs)
        output = unembedded_output.reshape(
            inputs.shape[0], # batch
            inputs.shape[1], # variate
            -1 # inferred to be seq_len * patch_size
        )
        
        return output