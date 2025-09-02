import pytest
import torch

@pytest.fixture
def test_data():
    """Provides a consistent set of test data for the modules."""
    batch_size = 2
    variates = 3
    time_steps = 100
    patch_size = 4
    embed_dim = 64
    stride = 4

    id_mask = id_mask = torch.zeros(batch_size, variates, time_steps, dtype=torch.long)
    id_mask[:, :, 48:] = 1

    return {
        "batch_size": batch_size,
        "variates": variates,
        "time_steps": time_steps,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "stride": stride, 
        "inputs": torch.rand(batch_size, variates, time_steps),
        
        "id_mask": id_mask, 
        "padded_inputs": torch.rand(2, 3, 10),
        "padded_mask": torch.tensor(
            [[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]],
            dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1)
    }