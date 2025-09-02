from toto.toto_backbone import TotoBackbone
from toto.scaler import StdMeanScaler

from tests.testing_data import test_data

def test_toto_backbone_forward(test_data):
    """Test the full forward pass of the TotoBackbone class."""
    
    # Set up model parameters

    
    patch_size = test_data["patch_size"]
    stride = test_data["patch_size"]
    embed_dim = test_data["embed_dim"]
    num_heads = 4
    mlp_hidden_dim = 256
    dropout = 0.1
    num_time_layers = 5
    num_space_layers = 1

    backbone = TotoBackbone(
        patch_size=patch_size,
        stride=stride,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim,
        dropout=dropout,
        num_time_layers=num_time_layers,
        num_space_layers=num_space_layers,
        scaler=StdMeanScaler(),
    )

    output = backbone(inputs=test_data["inputs"], id_mask=test_data["id_mask"])
    
    expected_output_shape = (test_data["batch_size"], test_data["variates"], test_data["time_steps"])
    assert output.shape == expected_output_shape

