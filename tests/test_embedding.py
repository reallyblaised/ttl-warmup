import pytest
import torch
import torch.nn as nn

from src.toto.embedding import PatchEmbedding, RotaryEmbedding
from src.toto.unembedding import Unembed

from tests.testing_data import test_data

## Embedding and Unembedding Tests ##
## ------------------------------- ##

def test_patch_embedding_forward(test_data):
    """Test the forward pass of the PatchEmbedding class."""
    patch_size = test_data["patch_size"]
    stride = test_data["stride"]
    embed_dim = test_data["embed_dim"]
    
    embed = PatchEmbedding(patch_size, stride, embed_dim)
    
    x_embed, id_mask_patched = embed.forward(test_data["inputs"], test_data["id_mask"])
    
    num_patches = test_data["time_steps"] // patch_size
    expected_shape = (test_data["batch_size"], test_data["variates"], num_patches, embed_dim)
    
    assert x_embed.shape == expected_shape
    assert id_mask_patched.shape == (test_data["batch_size"], test_data["variates"], num_patches)

def test_unembed_forward(test_data):
    """Test the forward pass of the Unembed class."""
    patch_size = test_data['patch_size']
    embed_dim = test_data['embed_dim']
    
    unembed = Unembed(embed_dim, patch_size)
    
    num_patches = test_data["time_steps"] // patch_size
    inputs = torch.rand(test_data["batch_size"], test_data["variates"], num_patches, embed_dim)
    
    output = unembed.forward(inputs)
    
    expected_shape = (test_data["batch_size"], test_data["variates"], test_data["time_steps"])
    
    assert output.shape == expected_shape


## Rotary Embedding Tests ## 
## ---------------------- ## 

@pytest.fixture
def rotary_emb():
    """
    Pytest fixture to create a RotaryEmbedding instance for testing.
    """
    head_dim = 16
    return RotaryEmbedding(head_dim)

def test_rotary_embedding_forward_pass_correct_rotation(rotary_emb):
    """
    Tests if the rotary embedding correctly applies the rotation to queries and keys.
    
    This test verifies the core mathematical logic of the rotation.
    We check that the rotation is applied as a 2D rotation for each pair of
    elements in the vector, which is the core principle of RoPE.
    """
    batch_size = 2
    num_heads = 4
    seq_len = 5
    head_dim = rotary_emb.head_dim

    # Create dummy queries and keys
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Get the expected sin and cos values for the first position
    cos = rotary_emb.cached_cos[..., 0, :].squeeze()
    sin = rotary_emb.cached_sin[..., 0, :].squeeze()

    # Apply the rotation to the first token manually
    q_first_token = q[..., 0, :]
    k_first_token = k[..., 0, :]
    
    # RoPE formula: x_rotated = x * cos + rotate_half(x) * sin
    q_rotated_expected = q_first_token * cos + _rotate_half(q_first_token) * sin
    k_rotated_expected = k_first_token * cos + _rotate_half(k_first_token) * sin
    
    # Get the rotated output from the module
    q_rotated_actual, k_rotated_actual = rotary_emb.rotate_queries_and_keys(q, k)
    
    # Compare the results for the first token
    assert torch.allclose(q_rotated_actual[..., 0, :], q_rotated_expected, atol=1e-6)
    assert torch.allclose(k_rotated_actual[..., 0, :], k_rotated_expected, atol=1e-6)

def _rotate_half(x):
    """
    Helper function to simulate the rotation for verification.
    """
    x_even, x_odd = x.chunk(2, dim=-1)
    return torch.cat((-x_odd, x_even), dim=-1)

def test_rotary_embedding_caching_and_recomputation(rotary_emb):
    """
    Tests if the module correctly handles caching and recomputation of angles.
    
    - It should not recompute for sequence lengths less than or equal to max_seq_len.
    - It should recompute and update max_seq_len for longer sequences.
    """
    # Test case 1: Sequence length is within the cached range
    seq_len_short = rotary_emb.max_seq_len // 2
    q = torch.randn(1, 1, seq_len_short, rotary_emb.head_dim)
    k = torch.randn(1, 1, seq_len_short, rotary_emb.head_dim)
    
    # Store initial cached values
    initial_cached_sin = rotary_emb.cached_sin.clone()
    initial_cached_cos = rotary_emb.cached_cos.clone()

    rotary_emb.rotate_queries_and_keys(q, k)
    
    # Caching should be unchanged
    assert torch.equal(rotary_emb.cached_sin, initial_cached_sin)
    assert torch.equal(rotary_emb.cached_cos, initial_cached_cos)

    # Test case 2: Sequence length exceeds the cached range
    seq_len_long = rotary_emb.max_seq_len + 5
    q = torch.randn(1, 1, seq_len_long, rotary_emb.head_dim)
    k = torch.randn(1, 1, seq_len_long, rotary_emb.head_dim)

    rotary_emb.rotate_queries_and_keys(q, k)
    
    # Caching should have been updated
    assert not torch.equal(rotary_emb.cached_sin, initial_cached_sin)
    assert not torch.equal(rotary_emb.cached_cos, initial_cached_cos)

    # The new cached tensors should have the new, longer sequence length
    assert rotary_emb.cached_sin.shape[2] == seq_len_long
    assert rotary_emb.cached_cos.shape[2] == seq_len_long
    assert rotary_emb.max_seq_len == seq_len_long

