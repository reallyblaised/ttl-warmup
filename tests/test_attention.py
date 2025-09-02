import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock

# Import the classes from the modules
from src.toto.attention import (
    TimeWiseMultiheadAttention, 
    SpaceWiseMultiheadAttention, 
    BaseMultiheadAttention,
    make_batched_block_mask, 
    make_space_mask,
    AttentionAxis
)
from src.toto.embedding import RotaryEmbedding

@pytest.fixture
def base_attention_time_axis():
    """Pytest fixture for a BaseMultiheadAttention instance with time-wise attention."""
    embed_dim = 256
    num_heads = 8
    dropout = 0.1
    rotary_emb = MagicMock(spec=RotaryEmbedding)
    use_memory_efficient_attention = False
    attention = TimeWiseMultiheadAttention(
        embed_dim,
        num_heads,
        dropout,
        rotary_emb,
        use_memory_efficient_attention,
    )
    return attention

@pytest.fixture
def base_attention_space_axis():
    """Pytest fixture for a BaseMultiheadAttention instance with space-wise attention."""
    embed_dim = 256
    num_heads = 8
    dropout = 0.1
    rotary_emb = MagicMock(spec=RotaryEmbedding)
    use_memory_efficient_attention = False
    attention = SpaceWiseMultiheadAttention(
        embed_dim,
        num_heads,
        dropout,
        rotary_emb,
        use_memory_efficient_attention,
    )
    
    return attention


## Testing _rearrange_inputs ##
## ------------------------- ##

def test_rearrange_inputs_time_axis(base_attention_time_axis):
    """Tests if the _rearrange_inputs method correctly reshapes the tensor for time-wise attention."""
    batch_size = 4
    variate = 5
    seq_len = 20
    embed_dim = base_attention_time_axis.embed_dim
    inputs = torch.randn(batch_size, variate, seq_len, embed_dim)
    rearranged = base_attention_time_axis._rearrange_inputs(inputs)
    
    # The new shape should be (batch_size * variate, seq_len, embed_dim)
    expected_shape = (batch_size * variate, seq_len, embed_dim)
    assert rearranged.shape == expected_shape

def test_rearrange_inputs_space_axis(base_attention_space_axis):
    """Tests if the _rearrange_inputs method correctly reshapes the tensor for space-wise attention."""
    batch_size = 4
    variate = 5
    seq_len = 20
    embed_dim = base_attention_space_axis.embed_dim
    inputs = torch.randn(batch_size, variate, seq_len, embed_dim)
    rearranged = base_attention_space_axis._rearrange_inputs(inputs)

    # The new shape should be (batch_size * seq_len, variate, embed_dim)
    expected_shape = (batch_size * seq_len, variate, embed_dim)
    assert rearranged.shape == expected_shape

def test_rearrange_inputs_time_axis_values(base_attention_time_axis):
    """Tests that tensor values are preserved correctly for time-wise attention."""
    # Use a simple tensor with a smaller embed_dim to easily verify values
    base_attention_time_axis.embed_dim = 1
    inputs = torch.arange(2 * 3 * 4).view(2, 3, 4, 1).float()
    rearranged = base_attention_time_axis._rearrange_inputs(inputs)

    # The elements should be concatenated along the batch and variate dimensions
    expected_values = torch.tensor([
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23
    ], dtype=torch.float32).view(6, 4, 1)
    
    assert torch.equal(rearranged, expected_values)

def test_rearrange_inputs_space_axis_values(base_attention_space_axis):
    """Tests that tensor values are preserved correctly for space-wise attention."""
    # Use a simple tensor with a smaller embed_dim to easily verify values
    base_attention_space_axis.embed_dim = 1
    inputs = torch.arange(2 * 3 * 4).view(2, 3, 4, 1).float()
    rearranged = base_attention_space_axis._rearrange_inputs(inputs)
    
    # The elements are permuted and then reshaped
    expected_permuted = inputs.permute(0, 2, 1, 3)
    expected_values = expected_permuted.reshape(8, 3, 1)

    assert torch.equal(rearranged, expected_values)

## Testing Space Mask Generation ##
## ----------------------------- ##

def test_make_space_mask_shape():
    """
    Tests if make_space_mask returns a tensor with the correct shape.
    """
    batch_size = 2
    variate = 3
    seq_len = 5
    id_mask = torch.randint(0, 10, (batch_size, variate, seq_len))

    mask = make_space_mask(id_mask)

    expected_shape = (batch_size * seq_len, 1, variate, variate)
    assert mask.shape == expected_shape

def test_make_space_mask_values_single_identity():
    """
    Tests if the mask values are correct for a single identity per batch.
    
    A single identity means all variates in a batch belong to the same group.
    The resulting mask should be all True.
    """
    batch_size = 2
    variate = 3
    seq_len = 5
    
    # Create an id_mask where all variates belong to the same identity (e.g., ID 1)
    id_mask = torch.ones(batch_size, variate, seq_len, dtype=torch.int)

    mask = make_space_mask(id_mask)

    # The mask should be all True, allowing all variates to attend to each other.
    assert torch.all(mask)

def test_make_space_mask_values_multiple_identities():
    """
    Tests if the mask values correctly block attention between different identities.
    
    This test creates a scenario with two distinct identities.
    """
    batch_size = 1
    variate = 4
    seq_len = 2
    
    # Create an id_mask with two identities
    # Variates 0 and 1 belong to identity 10
    # Variates 2 and 3 belong to identity 20
    # The mask for a single sequence step should look like:
    # [[T, T, F, F],
    #  [T, T, F, F],
    #  [F, F, T, T],
    #  [F, F, T, T]]
    id_mask = torch.tensor([
        [[10, 10], [10, 10], [20, 20], [20, 20]]
    ], dtype=torch.int)

    mask = make_space_mask(id_mask)

    expected_mask_per_seq_step = make_batched_block_mask(id_mask[0, :, 0])
    
    # The mask should be the same for each sequence step
    # Reshape the output mask to be easier to compare
    reshaped_mask = mask.view(seq_len, variate, variate)

    # Check if the mask for the first sequence step is correct
    assert torch.equal(reshaped_mask[0], expected_mask_per_seq_step)

    # Check if the mask for the second sequence step is also correct
    assert torch.equal(reshaped_mask[1], expected_mask_per_seq_step)


## Attention and Transformer Tests ##
## ------------------------------- ##

def test_make_batched_block_mask():
    """Test the make_batched_block_mask function."""
    id_tensor = torch.tensor([[0, 0, 1, 1, 2, 2]])
    mask = make_batched_block_mask(id_tensor)
    
    expected_mask = torch.tensor([
        [True, True, False, False, False, False],
        [True, True, False, False, False, False],
        [False, False, True, True, False, False],
        [False, False, True, True, False, False],
        [False, False, False, False, True, True],
        [False, False, False, False, True, True],
    ])
    
    assert torch.equal(mask.squeeze(0), expected_mask)

## Space Attention Tests ##
## --------------------- ##

def test_spacewise_attention_forward_pass():
    """
    Tests the forward pass of the SpaceWiseMultiheadAttention module.
    
    This test verifies that the module produces an output of the correct shape
    and that the attention mask correctly restricts attention. We do this by
    creating two distinct identity groups and ensuring that the output for 
    one group is not affected by changes in the input of the other.
    """
    embed_dim = 16
    num_heads = 2
    dropout = 0.0
    rotary_emb = MagicMock(spec=RotaryEmbedding)
    use_memory_efficient_attention = False

    # Instantiate the SpaceWiseMultiheadAttention module
    attention = SpaceWiseMultiheadAttention(
        embed_dim,
        num_heads,
        dropout,
        rotary_emb,
        use_memory_efficient_attention,
    )
    
    # Define input parameters
    batch_size = 1
    variate = 4
    seq_len = 2
    
    # Create an identity mask with two groups: variates 0,1 and variates 2,3
    id_mask = torch.tensor([
        [[10, 10], [10, 10], [20, 20], [20, 20]]
    ], dtype=torch.int)

    # Create the corresponding attention mask
    attn_mask = make_space_mask(id_mask)

    # Create input tensor where the second group (variates 2,3) has a distinct value
    inputs_group_a = torch.ones(batch_size, variate, seq_len, embed_dim)
    
    # Change the values for the second group to a different value
    inputs_group_a[:, 2:, :, :] = 2.0

    # Pass the input through the attention module
    output_a = attention.forward(inputs_group_a, attention_mask=attn_mask)
    
    # Create a second input tensor where we only change the values for the
    # second group, which should be masked out for the first group's attention
    inputs_group_b = inputs_group_a.clone()
    inputs_group_b[:, 2:, :, :] = 3.0

    # Pass the second input through the attention module
    output_b = attention.forward(inputs_group_b, attention_mask=attn_mask)

    # Assert that the output shape is correct
    expected_shape = (batch_size, variate, seq_len, embed_dim)
    assert output_a.shape == expected_shape
    assert output_b.shape == expected_shape

    # Check if the output for the first group (variates 0,1) is identical
    # in both cases, as it should not be affected by changes in the second group
    assert torch.equal(output_a[:, :2, :, :], output_b[:, :2, :, :])

    # Check that the outputs for the second group (variates 2,3) are different
    # because their inputs were different
    assert not torch.equal(output_a[:, 2:, :, :], output_b[:, 2:, :, :])


## Time Attention Tests ##
## -------------------- ##`



@pytest.fixture
def time_attention():
    """Pytest fixture for a TimeWiseMultiheadAttention instance."""
    embed_dim = 16
    num_heads = 2
    dropout = 0.0
    rotary_emb = RotaryEmbedding(head_dim=2)
    use_memory_efficient_attention = False
    return TimeWiseMultiheadAttention(
        embed_dim,
        num_heads,
        dropout,
        rotary_emb,
        use_memory_efficient_attention,
    )

def test_timewise_attention_forward_pass_shape(time_attention):
    """
    Tests if the TimeWiseMultiheadAttention forward pass produces a tensor
    with the correct output shape.
    """
    batch_size = 2
    variate = 3
    seq_len = 5
    embed_dim = time_attention.embed_dim

    inputs = torch.randn(batch_size, variate, seq_len, embed_dim)
    output = time_attention.forward(inputs)

    expected_shape = (batch_size, variate, seq_len, embed_dim)
    assert output.shape == expected_shape

def test_timewise_attention_forward_pass_causality(time_attention):
    """
    Tests if the time-wise attention mechanism is causal.
    
    This test verifies that the output for a given time step depends only
    on the input from the current and previous time steps, and not on future
    time steps. We do this by modifying a future time step and ensuring
    that the output for a preceding time step remains unchanged.
    """
    batch_size = 1
    variate = 1
    seq_len = 5
    embed_dim = time_attention.embed_dim

    # Create a base input tensor
    inputs_a = torch.randn(batch_size, variate, seq_len, embed_dim)

    # Pass the input through the attention module
    output_a = time_attention.forward(inputs_a)

    # Create a second input tensor where we only change the last time step
    inputs_b = inputs_a.clone()
    inputs_b[:, :, -1, :] = torch.randn_like(inputs_b[:, :, -1, :])

    # Pass the second input through the attention module
    output_b = time_attention.forward(inputs_b)

    # The output for all time steps except the last should be identical
    # because of the causal mask.
    assert torch.equal(output_a[:, :, :-1, :], output_b[:, :, :-1, :])

    # The output for the last time step should be different because its
    # input was modified.
    assert not torch.equal(output_a[:, :, -1, :], output_b[:, :, -1, :])


def test_timewise_attention_forward_pass_causality_multivariate(time_attention):
    """
    Tests causality for a multivariate input to the time-wise attention.
    
    This verifies that changing a future time step of one variate does not affect
    the output of other variates.
    """
    batch_size = 1
    variate = 3  # Test with multiple variates
    seq_len = 5
    embed_dim = time_attention.embed_dim

    # Create a base input tensor with multiple variates
    inputs_a = torch.randn(batch_size, variate, seq_len, embed_dim)

    # Pass the input through the attention module
    output_a = time_attention.forward(inputs_a)

    # Create a second input tensor where only the last time step of the first variate is changed.
    inputs_b = inputs_a.clone()
    inputs_b[:, 0, -1, :] = torch.randn_like(inputs_b[:, 0, -1, :])

    # Pass the second input through the attention module
    output_b = time_attention.forward(inputs_b)
    
    # Assert that the output for the *other* variates remains unchanged.
    assert torch.equal(output_a[:, 1:, :, :], output_b[:, 1:, :, :])

    # Assert that the output for the first variate up to the second-to-last
    # time step is unchanged due to causality.
    assert torch.equal(output_a[:, 0, :-1, :], output_b[:, 0, :-1, :])
    
    # Assert that the output for the last time step of the first variate has changed.
    assert not torch.equal(output_a[:, 0, -1, :], output_b[:, 0, -1, :])




