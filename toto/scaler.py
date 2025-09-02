import warnings
from abc import ABC, abstractmethod
from typing import Tuple, Type

import torch
import torch.nn as nn 


class Scaler(ABC, torch.nn.Module):
    """
    Abstract Base Class for a data scaler. All scaler implementations must
    inherit from this class and implement the forward method.
    """

    @abstractmethod
    def forward(
        self,
        data: torch.Tensor,
        padding_mask: torch.Tensor,
        weights: torch.Tensor,
        prefix_length: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Applies a scaling transformation to the input data.

        Parameters
        ----------
        data
            The input data tensor of shape [batch, variate, time_steps].
        padding_mask
            A boolean mask indicating padded values in the input tensor.
        weights
            A tensor of weights for the data points.
        prefix_length
            Optional prefix length for causal computations.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - The scaled data tensor.
            - The location (mean) used for scaling.
            - The scale (standard deviation) used for scaling.
        """
        pass



class StdMeanScaler(Scaler):
    """
    Scales data to have zero mean and unit variance using global statistics
    (i.e., over all data points provided).
    """

    def __init__(
        self,
        dim: int = -1,
        keepdim: bool = True,
        minimum_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim
        self.minimum_scale = minimum_scale

    def forward(
        self,
        data: torch.Tensor,
        padding_mask: torch.Tensor,
        weights: torch.Tensor,
        prefix_length: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scales the input data to have zero mean and unit variance over the whole 
        time period.

        This method calculates the weighted mean and standard deviation of the input 
        `data` along the specified dimension (`self.dim`). It handles masked and 
        weighted data to ensure the statistics are computed correctly.

        Args:
            data (torch.Tensor): The input data tensor to be scaled.
            padding_mask (torch.Tensor): A boolean or float tensor where 1s indicate 
                valid data points and 0s indicate padding. This mask is used to
                exclude padded values from the statistics calculation.
            weights (torch.Tensor): A tensor of importance weights for each data 
                point. This allows certain data points to contribute more to the 
                calculated statistics.
            prefix_length (int | None): An optional integer specifying the length 
                of a prefix to consider. If provided, the statistics will only be 
                calculated on data up to this length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The scaled data tensor.
                - The mean (`loc`) used for scaling.
                - The standard deviation (`scale`) used for scaling.
        """
        assert data.shape == weights.shape, "data and weights must have same shape"
        
        with torch.no_grad():
            if prefix_length is not None:
                prefix_mask = torch.zeros_like(weights)
                prefix_mask[..., :prefix_length] = 1.0
                weights = weights * prefix_mask

            weights = weights * padding_mask

            # Simplified calculation using float32
            denominator = weights.sum(self.dim, keepdim=self.keepdim).clamp_min(1.0)
            means = (data * weights).sum(self.dim, keepdim=self.keepdim) / denominator
            means = torch.nan_to_num(means)

            variance = (((data - means) * weights) ** 2).sum(
                self.dim, keepdim=self.keepdim
            ) / denominator
            scale = torch.sqrt(variance + self.minimum_scale)
            loc = means

            return (data - loc) / scale, loc, scale




class CausalStdMeanScaler(Scaler):
    """
    Causally scales the data along the time dimension using statistics
    computed only from past and present data points.
    """
    def __init__(
        self,
        dim: int = -1,
        minimum_scale: float = 0.1,
        use_bessel_correction: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.minimum_scale = minimum_scale
        self.use_bessel_correction = use_bessel_correction
        
    def forward(
        self,
        data: torch.Tensor,
        padding_mask: torch.Tensor,
        weights: torch.Tensor,
        prefix_length: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Scales the input data to have zero mean and unit variance. It does so in 
        a consistent point in time manner.

        This method calculates the weighted mean and standard deviation of the input 
        `data` along the specified dimension (`self.dim`). It handles masked and 
        weighted data to ensure the statistics are computed correctly.

        Args:
            data (torch.Tensor): The input data tensor to be scaled.
            padding_mask (torch.Tensor): A boolean or float tensor where 1s indicate 
                valid data points and 0s indicate padding. This mask is used to
                exclude padded values from the statistics calculation.
            weights (torch.Tensor): A tensor of importance weights for each data 
                point. This allows certain data points to contribute more to the 
                calculated statistics.
            prefix_length (int | None): An optional integer specifying the length 
                of a prefix to consider. If provided, the statistics will only be 
                calculated on data up to this length.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The scaled data tensor.
                - The mean (`loc`) used for scaling.
                - The standard deviation (`scale`) used for scaling.
        """
        
        assert data.shape == weights.shape, "data and weights must have same shape"
        assert len(data.shape) == 3, "Input data must have shape [batch, variates, time_steps]"
        assert self.dim == -1, "CausalStdMeanScaler only supports dim=-1 (last dimension)"
        
        with torch.no_grad():
            weights = weights * padding_mask
            
            # Simplified calculation using cumsum and float32
            cum_weights = torch.cumsum(weights, dim=self.dim)
            cum_values = torch.cumsum(data * weights, dim=self.dim)
            
            denominator = cum_weights.clamp_min(1.0)
            causal_means = cum_values / denominator
            
            causal_variance = torch.cumsum((data - causal_means)**2 * weights, dim=self.dim) / denominator
            causal_scale = torch.sqrt(causal_variance + self.minimum_scale)
            
            scaled_data = (data - causal_means) / causal_scale
            
            return scaled_data, causal_means, causal_scale