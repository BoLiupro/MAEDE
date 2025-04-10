import torch
import math
from torch import nn
from torch.nn.functional import normalize

class PositionalEncoding(nn.Module):
    """spatial-temporal positional encoding."""

    def __init__(self, spatial_dim, temporal_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.spatial_position_embedding = nn.Linear(spatial_dim, hidden_dim)
        self.temporal_position_embedding = nn.Linear(temporal_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

        # self.register_buffer(
        #     "temporal_position",
        #     self._time_embedding(t_max_len, temporal_dim),
        #     persistent=False,
        # )

    def forward(self, spe, tpe):
        """Positional encoding

        Args:
            spe (torch.Tensor): [B, N, d_s] spatial embeddings.
            tpe (torch.Tensor): [B, L, d_t] temporal embeddings.
        Returns:
            spe (torch.tensor): [B, N, d_s]
            tpe (torch.tensor): [B, L, d_t]
        """
        if len(spe.shape) == 2:
            spe = spe.unsqueeze(0)
        if len(tpe.shape) == 2:
            tpe = tpe.unsqueeze(0)

        # temporal positional encoding
        tpe = torch.relu(self.temporal_position_embedding(tpe))
        tpe = self.dropout(tpe)

        # spatial positional encoding
        spe = torch.relu(self.spatial_position_embedding(spe))
        spe = self.dropout(spe)

        return spe, tpe
