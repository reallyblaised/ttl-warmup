import pytest
import torch
import torch.nn as nn

# Import the classes from the modules

from toto.normalise import RMSNorm
from toto.scaler import StdMeanScaler, CausalStdMeanScaler
from tests.testing_data import test_data

## Scaler Tests ##
## ------------ ##

def test_std_mean_scaler_forward(test_data):
    """Test the StdMeanScaler class."""
    scaler = StdMeanScaler(dim=-1)
    scaled_data, loc, scale = scaler(test_data["inputs"], torch.ones_like(test_data["inputs"]), torch.ones_like(test_data["inputs"]))

    assert scaled_data.shape == test_data["inputs"].shape
    assert loc.shape == (test_data["batch_size"], test_data["variates"], 1)
    assert scale.shape == (test_data["batch_size"], test_data["variates"], 1)
    
    # Check that the scaled data has a mean close to 0 and std close to 1
    assert torch.allclose(scaled_data.mean(dim=-1), torch.zeros_like(scaled_data.mean(dim=-1)), atol=1e-6)
    assert torch.allclose(scaled_data.std(dim=-1), torch.ones_like(scaled_data.std(dim=-1)), atol=1e-2)

    
def test_causal_std_mean_scaler_forward(test_data):
    """Test the CausalStdMeanScaler class."""
    scaler = CausalStdMeanScaler()
    scaled_data, _, _ = scaler(
        data=test_data["padded_inputs"],
        padding_mask=test_data["padded_mask"],
        weights=torch.ones_like(test_data["padded_inputs"])
    )

    assert scaled_data.shape == test_data["padded_inputs"].shape
    
    # Check that the first element of the scaled data is approximately 0
    # since it's scaled by its own mean and std.
    assert torch.allclose(scaled_data[:, :, 0], torch.zeros_like(scaled_data[:, :, 0]), atol=1e-6)

## RMSNorm Test ##
## ------------ ##

    
@pytest.fixture
def input_tensor():
    """Provides a sample input tensor for the tests."""
    return torch.randn(2, 3, 128)

@pytest.mark.parametrize("dim", [64, 128, 256])
def test_output_shape(dim, input_tensor):
    """Tests that RMSNorm maintains the correct output shape."""
    # Create a new tensor with the parameterized dimension
    x = torch.randn(2, 3, dim)
    norm = RMSNorm(dim)
    output = norm(x)
    assert output.shape == x.shape, "Output shape should be the same as input shape."

def test_normalization_properties(input_tensor):
    """
    Tests that the output of RMSNorm has an RMS of approximately 1.
    This is the core property of the normalization.
    """
    dim = input_tensor.shape[-1]
    norm = RMSNorm(dim)
    output = norm(input_tensor)
    
    # After normalization, the RMS of the last dimension should be close to 1.
    rms_output = torch.sqrt(torch.mean(output * output, dim=-1))
    # We use `torch.allclose` for floating-point comparisons.
    assert torch.allclose(rms_output, torch.ones_like(rms_output), atol=1e-3), \
           "RMS of the output tensor should be close to 1."

def test_epsilon_parameter():
    """Tests that the epsilon parameter prevents division by zero."""
    # Create a tensor of zeros to force the normalization to use epsilon.
    x = torch.zeros(1, 10)
    norm = RMSNorm(10, eps=1e-6)
    
    # No error should be raised when performing the forward pass.
    try:
        norm(x)
    except Exception as e:
        pytest.fail(f"RMSNorm failed to handle a zero input tensor: {e}")


