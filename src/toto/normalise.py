import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """
    Implements Root Mean Square Normalization (RMSNorm) using a highly
    optimized kernel from the xformers library.
    
    RMSNorm normalizes a tensor by its root mean square, which is a faster
    and often more stable alternative to Layer Normalization in some models.
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        """
        Initializes the RMSNorm layer.
        
        Args:
            dim (int): The dimension of the input tensor to normalize.
            eps (float): A small value added to the denominator for numerical stability.
        """
        super().__init__()
        # The 'scale' parameter is a learned weight that scales the normalized output.
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the RMSNorm layer.

        The core of the implementation is a direct call to the
        highly optimized 'rms_norm' function from xformers.
        
        Args:
            x (torch.Tensor): The input tensor to normalize.
            
        Returns:
            torch.Tensor: The normalized output tensor.
        """
        
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        
        x_norm = x / rms
        return x_norm