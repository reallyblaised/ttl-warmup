import torch
import torch.nn as nn 


from src.toto.embedding import PatchEmbedding
from src.toto.scaler import Scaler
from src.toto.toto_transformer import TotoTransformer
from src.toto.unembedding import Unembed

class TotoBackbone(nn.Module):
    """
    A simplified version of the TotoBackbone class,
    demonstrating the logical flow of data through the model.
    """

    def __init__(
        self,
        patch_size: int,
        stride: int,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        dropout: float, 
        num_time_layers: int,
        num_space_layers: int,
        scaler: Scaler, # First lets create the Scaler Class 
        # output_distribution: DistributionOutput,
    ):
        super().__init__()
        self.scaler = scaler
        self.patch_embed = PatchEmbedding(patch_size, stride, embed_dim)
        self.transformer = TotoTransformer( # We might need some of these blocks. 
            embed_dim, num_heads, mlp_hidden_dim, dropout, num_time_layers, num_space_layers
        ) 

        self.unembed = Unembed(embed_dim, patch_size)

    def forward(self, inputs, id_mask):
        ones = torch.ones_like(inputs)
        scaled_inputs, mean, std = self.scaler(inputs, 
                                               padding_mask = ones,
                                               weights = ones) # For the time being keep it simple. 
        patched_inputs, patched_id_mask = self.patch_embed(scaled_inputs, id_mask)

        hidden_states = self.transformer(patched_inputs, patched_id_mask)
        unscaled_out = self.unembed(hidden_states) 

        scaled_out  = unscaled_out * std + mean
        return scaled_out

