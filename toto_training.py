import torch
import torch.nn as nn
from toto.toto_backbone import TotoBackbone
from toto.scaler import StdMeanScaler
from tests.testing_data import test_data

def train_loop(model, inputs, id_mask, loss_function, optimizer, epochs=10):
    """
    Performs a training loop for the TOTO model.
    """
    model.train()  # Set the model to training mode

    for epoch in range(epochs):
        # Reset gradients for each epoch
        optimizer.zero_grad()
        
        # Forward pass: get model prediction
        # The model's output is expected to have the same shape as the input.
        predictions = model(inputs=inputs, id_mask=id_mask)
        
        # Calculate the loss between predictions and original inputs.
        # This is a self-supervised approach where the model predicts the input.
        loss = loss_function(predictions, inputs)
        
        # Backward pass: compute gradients and update weights
        loss.backward()
        optimizer.step()
        
        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 4. Run the training loop with the sample data.


if __name__ == '__main__':
    model = TotoBackbone(
    patch_size=None,
    stride=None,
    embed_dim=None,
    num_heads=4,
    mlp_hidden_dim=256,
    dropout=0.1,
    num_time_layers=5,
    num_space_layers=1,
    scaler=StdMeanScaler(),
    )
    
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if torch.backends.mps.is_available():
    # If the MPS backend is available, set the device to 'mps'.
        device = torch.device("mps")
        print("Apple GPU (MPS) is available.")
    else:
        # If not, set the device to 'cpu' as a fallback.
        device = torch.device("cpu")
        print("Apple GPU (MPS) is not available. Using CPU.")

    model.to(device)
    inputs.to(device)
    id_mask.to(device)
    
    print("Starting training loop...")
    train_loop(model, inputs, id_mask, loss_function, optimizer)
    print("Training finished.")
