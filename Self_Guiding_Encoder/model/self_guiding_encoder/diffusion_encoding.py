import torch
from torch import nn
import torch.nn.functional as F


class DiffusionEncoding(nn.Module):
    """Diffusion step encoding"""
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None, dropout=0.1):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),  # (T,dim*2)
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, diffusion_step):
        """Positional encoding

        Args:
            diffusion_step (torch.Tensor): [B] diffusion step for each sample in the batch.
        Returns:
            torch.tensor: output sequence
        """
        emd = self.embedding[diffusion_step]
        emd = self.projection1(emd)
        emd = F.silu(emd)  # [B, emd_dim]
        emd = self.dropout(emd)
        return emd

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


# class DiffusionEncoding(nn.Module):
#     """Diffusion step encoding"""
#     def __init__(self, num_steps, embedding_dim, projection_dim=None, dropout=0.1):
#         super().__init__()
#         diff_dim = 128
#         if projection_dim is None:
#             projection_dim = [embedding_dim, embedding_dim]
#         self.projection_dim = projection_dim
#         self.register_buffer(
#             "embedding",
#             self._build_embedding(num_steps, diff_dim / 2),  # (T,dim*2)
#             persistent=False,
#         )
#         self.projection1 = nn.Linear(diff_dim, int(diff_dim/2))
#         self.projection2 = nn.Linear(int(diff_dim/2), int(diff_dim/4))
#         self.projection3 = nn.Linear(int(diff_dim/4), projection_dim[0] * projection_dim[1])
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, diffusion_step):
#         """Positional encoding
#
#         Args:
#             input_data (torch.tensor): [B, N, L, d_x] input sequence for temporal position learning.
#             diffusion_step (torch.Tensor): [B] diffusion step for each sample in the batch.
#         Returns:
#             torch.tensor: output sequence
#         """
#         emd = self.embedding[diffusion_step]
#         emd = self.dropout(F.silu(self.projection1(emd)))
#         emd = self.dropout(F.silu(self.projection2(emd)))
#         emd = self.dropout(F.silu(self.projection3(emd)))
#         emd = emd.view(-1, self.projection_dim[0], self.projection_dim[1]) # [B, p_dim[0], p_dim[1]]
#         return emd
#
#     def _build_embedding(self, num_steps, dim=64):
#         steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
#         frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
#         table = steps * frequencies  # (T,dim)
#         table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
#         return table