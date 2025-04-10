import torch
from torch import nn


class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self):
        super().__init__()

    def forward(self, input_data, mask_ratio):
        """
        spatial mask, temporal mask, noise mask
        Args:
            input_data (torch.Tensor): input with shape [B, N, L, D].
            mask_ratio (float): mask ratio.
        Returns:
            torch.Tensor: make index with shape [B, N, L]. 1 means masked, 0 means unmasked.
        """
        batch_size, num_nodes, t_len, _ = input_data.shape
        mask = torch.rand(batch_size, num_nodes, t_len, device=input_data.device) < mask_ratio
        # mask = torch.rand(1, num_nodes, 1, device=input_data.device) < mask_ratio
        # mask = mask.repeat(batch_size, 1, t_len)
        return mask.float()
