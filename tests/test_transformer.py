import torch
import pytest
from src.toto.toto_transformer import SwiGLU, TotoTransformerLayer, TotoTransformer
from src.toto.attention import AttentionAxis, make_space_mask
from src.toto.embedding import RotaryEmbedding

# --- Fixtures ---
@pytest.fixture
def swiglu_instance():
    """Fixture for SwiGLU module."""
    in_features = 16
    hidden_features = 32
    return SwiGLU(in_features, hidden_features)

@pytest.fixture
def transformer_layer_time():
    """Fixture for a TimeWiseMultiheadAttention layer."""
    embed_dim = 16
    num_heads = 2
    mlp_hidden_dim = 32
    dropout = 0.0
    rotary_emb = RotaryEmbedding(head_dim=embed_dim // num_heads)
    attention_axis = AttentionAxis.TIME
    return TotoTransformerLayer(
        embed_dim, num_heads, mlp_hidden_dim, dropout, rotary_emb, attention_axis
    )

@pytest.fixture
def transformer_layer_space():
    """Fixture for a SpaceWiseMultiheadAttention layer."""
    embed_dim = 16
    num_heads = 2
    mlp_hidden_dim = 32
    dropout = 0.0
    rotary_emb = None
    attention_axis = AttentionAxis.SPACE
    return TotoTransformerLayer(
        embed_dim, num_heads, mlp_hidden_dim, dropout, rotary_emb, attention_axis
    )

@pytest.fixture
def toto_transformer_instance():
    """Fixture for the main TotoTransformer module."""
    embed_dim = 16
    num_heads = 2
    mlp_hidden_dim = 32
    dropout = 0.0
    num_time_layers = 2
    num_space_layers = 2
    return TotoTransformer(
        embed_dim, num_heads, mlp_hidden_dim, dropout, num_time_layers, num_space_layers
    )

## SwiGLU Tests ##
## ------------ ## 

def test_swiglu_forward_pass(swiglu_instance):
    """Test SwiGLU module's forward pass shape."""
    in_features = swiglu_instance.linear_gate.in_features
    batch_size, seq_len = 4, 10
    x = torch.randn(batch_size, seq_len, in_features)
    output = swiglu_instance.forward(x)
    assert output.shape == (batch_size, seq_len, in_features)

## TotoTransformerLayer Tests ##
## -------------------------- ##

def test_transformer_layer_time_forward_pass_shape(transformer_layer_time):
    """Test TotoTransformerLayer with time attention forward pass shape."""
    batch_size, variate, seq_len, embed_dim = 2, 5, 8, transformer_layer_time.attention.embed_dim
    inputs = torch.randn(batch_size, variate, seq_len, embed_dim)
    output = transformer_layer_time.forward(inputs)
    assert output.shape == inputs.shape

def test_transformer_layer_space_forward_pass_shape(transformer_layer_space):
    """Test TotoTransformerLayer with space attention forward pass shape."""
    batch_size, variate, seq_len, embed_dim = 2, 5, 8, transformer_layer_space.attention.embed_dim
    inputs = torch.randn(batch_size, variate, seq_len, embed_dim)
    # Mock a space mask
    attention_mask = torch.ones(batch_size * seq_len, 1, variate, variate, dtype=torch.bool)
    output = transformer_layer_space.forward(inputs, attention_mask=attention_mask)
    assert output.shape == inputs.shape

## TotoTransformer (Decoder Block) Tests ##
## ------------------------------------- ##

def test_toto_transformer_forward_pass_shape(toto_transformer_instance):
    """Test the full TotoTransformer forward pass shape."""
    batch_size, variate, seq_len, embed_dim = 2, 5, 8, toto_transformer_instance.time_transformers[0].attention.embed_dim
    inputs = torch.randn(batch_size, variate, seq_len, embed_dim)
    
    # Create a simple identity mask
    id_mask = torch.randint(0, 5, (batch_size, variate, seq_len))
    
    output = toto_transformer_instance.forward(inputs, id_mask)
    assert output.shape == inputs.shape

def test_toto_transformer_time_attend_causality(toto_transformer_instance):
    """
    Tests that the time attention part of the TotoTransformer is causal.
    
    This test verifies that a change in a future time step does not affect the
    output of a previous time step.
    """
    batch_size, variate, seq_len, embed_dim = 1, 1, 5, toto_transformer_instance.time_transformers[0].attention.embed_dim
    inputs_a = torch.randn(batch_size, variate, seq_len, embed_dim)
    
    # Pass input A through the time-wise attention layers
    output_a = toto_transformer_instance._time_attend(inputs_a)

    # Create a second input B where only the last time step is changed
    inputs_b = inputs_a.clone()
    inputs_b[:, :, -1, :] = torch.randn_like(inputs_b[:, :, -1, :])
    
    # Pass input B through the time-wise attention layers
    output_b = toto_transformer_instance._time_attend(inputs_b)

    # Assert that the output for the first four time steps is the same
    assert torch.equal(output_a[:, :, :-1, :], output_b[:, :, :-1, :])

    # Assert that the output for the last time step is different
    assert not torch.equal(output_a[:, :, -1, :], output_b[:, :, -1, :])

def test_toto_transformer_time_attend_causality_multivariate(toto_transformer_instance):
    """
    Tests that the time attention part of the TotoTransformer is causal
    for a multivariate input (variate > 1).
    
    This test verifies that changing a future time step of one variate does not affect
    the output of other variates or past time steps.
    """
    batch_size = 1
    variate = 3  # Test with multiple variates
    seq_len = 5
    embed_dim = toto_transformer_instance.time_transformers[0].attention.embed_dim
    
    # Create a base input tensor with multiple variates
    inputs_a = torch.randn(batch_size, variate, seq_len, embed_dim)
    
    # Pass input A through the time-wise attention layers
    output_a = toto_transformer_instance._time_attend(inputs_a)
    
    # Create a second input B where only the last time step of the first variate is changed.
    inputs_b = inputs_a.clone()
    inputs_b[:, 0, -1, :] = torch.randn_like(inputs_b[:, 0, -1, :])
    
    # Pass input B through the time-wise attention layers
    output_b = toto_transformer_instance._time_attend(inputs_b)
    
    # Assert that the output for the *other* variates (variates 1 and 2) remains unchanged.
    assert torch.equal(output_a[:, 1:, :, :], output_b[:, 1:, :, :])
    
    # Assert that the output for the first variate up to the second-to-last
    # time step is unchanged due to causality.
    assert torch.equal(output_a[:, 0, :-1, :], output_b[:, 0, :-1, :])
    
    # Assert that the output for the last time step of the first variate has changed.
    assert not torch.equal(output_a[:, 0, -1, :], output_b[:, 0, -1, :])

    
def test_toto_transformer_space_attend_masking(toto_transformer_instance):
    """
    Tests that the space attention part of the TotoTransformer respects the mask.
    
    This test creates two distinct identity groups and verifies that a change
    in one group's input does not affect the output of the other.
    """
    batch_size = 1
    variate = 4
    seq_len = 1  # We can use a small sequence length for simplicity
    embed_dim = toto_transformer_instance.time_transformers[0].attention.embed_dim

    # Create an identity mask with two groups
    # Group 1 (variates 0 and 1) have ID 10
    # Group 2 (variates 2 and 3) have ID 20
    id_mask = torch.tensor([
        [[10], [10], [20], [20]]
    ], dtype=torch.int)

    # Manually create the correct attention mask for the test scenario
    # We call make_space_mask directly with the required arguments.
    attention_mask = make_space_mask(id_mask)
    
    # Create a base input tensor for both tests
    inputs_a = torch.randn(batch_size, variate, seq_len, embed_dim)

    # Get the initial output for a clean run
    output_a = toto_transformer_instance._space_attend(inputs_a, attention_mask)

    # Create a new input tensor with a difference only in the second group (masked out)
    inputs_b = inputs_a.clone()
    inputs_b[:, 2:, :, :] = torch.randn_like(inputs_b[:, 2:, :, :])
    
    # Run the second input through the space attention layers
    output_b = toto_transformer_instance._space_attend(inputs_b, attention_mask)
    
    # Check that the output for the first group (variates 0, 1) is identical
    # in both runs, as their attention should not be affected by the second group's input
    assert torch.equal(output_a[:, :2, :, :], output_b[:, :2, :, :])
    
    # Check that the output for the second group (variates 2, 3) is different
    # because their input was modified, and they are not masked from each other
    assert not torch.equal(output_a[:, 2:, :, :], output_b[:, 2:, :, :])

