import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn import MultiheadAttention


class SpatialTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim

        self.dpe_layer = nn.Linear(hidden_dim, hidden_dim)
        self.spe_layer = nn.Linear(hidden_dim, hidden_dim)
        self.tpe_layer = nn.Linear(hidden_dim, hidden_dim)

        self.ca_layer = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.sa_layer = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)

        # fusion layer
        self.fc_ca = nn.Linear(hidden_dim, hidden_dim)
        self.fc_sa = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, src, condition, side_info, training=True):
        """

        Args:
            src: [B, N, d]
            condition: [B, N, d]
            side_info: Dict()
        Returns:

        """
        spe = torch.relu(self.spe_layer(side_info['spe']))
        dpe = torch.relu(self.dpe_layer(side_info['dpe']))
        context_spe = spe[:, side_info['context_index']]
        target_spe = spe[:, side_info['target_index']]
        src_attn = src + dpe.unsqueeze(1)

        # spatial transformer
        src_attn = src_attn + target_spe
        ca_src = src_attn.transpose(0, 1)  # [N, B, D]
        ca_src_kv = condition + context_spe
        ca_src_kv = ca_src_kv.transpose(0, 1)
        ca_src = self.ca_layer(ca_src, ca_src_kv, ca_src_kv, src_mask=None)
        ca_src = ca_src.transpose(0, 1)

        sa_src = src_attn.transpose(0, 1)
        sa_src = self.sa_layer(sa_src, sa_src, sa_src, src_mask=None)
        sa_src = sa_src.transpose(0, 1)

        # fusion
        fusion = torch.sigmoid(self.fc_ca(ca_src) + self.fc_sa(sa_src) + dpe.unsqueeze(1) + target_spe)
        s_src = fusion * ca_src + (1 - fusion) * sa_src
        return s_src

class TemporalTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, mlp_ratio, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = hidden_dim

        self.dpe_layer = nn.Linear(hidden_dim, hidden_dim)
        self.tpe_layer = nn.Linear(hidden_dim, hidden_dim)
        self.spe_layer = nn.Linear(hidden_dim, hidden_dim)
        self.ca_layer = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)
        self.sa_layer = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim*mlp_ratio, dropout)

        # fusion layer
        self.fc_ca = nn.Linear(hidden_dim, hidden_dim)
        self.fc_sa = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, src, condition, side_info, training=True):
        """

        Args:
            src: [B, N, d]
            condition: [B, N, L, d]
            side_info: {} side information
        Returns:

        """
        B, N, D = src.shape
        L = condition.shape[2]

        tpe = torch.relu(self.tpe_layer(side_info['tpe']))
        spe = torch.relu(self.spe_layer(side_info['spe']))
        dpe = torch.relu(self.dpe_layer(side_info['dpe']))
        src_attn = src + dpe.unsqueeze(1) + tpe[:, L:L+1, :] + spe

        # temporal transformer
        t_src_kv = condition + dpe.unsqueeze(1).unsqueeze(1) + tpe[:, :L].unsqueeze(1) + spe.unsqueeze(2)
        t_src_kv = t_src_kv.reshape(B*N, L, D).transpose(0, 1)
        ca_src = src_attn.unsqueeze(2).reshape(B*N, 1, D).transpose(0, 1)
        ca_src = self.ca_layer(ca_src, t_src_kv, t_src_kv)
        ca_src = ca_src.transpose(0, 1).reshape(B, N, D)

        sa_src = src_attn.transpose(0, 1)
        sa_src = self.sa_layer(sa_src, sa_src, sa_src)
        sa_src = sa_src.transpose(0, 1)

        # fusion layer
        fusion = torch.sigmoid(self.fc_ca(ca_src) + self.fc_sa(sa_src) + dpe.unsqueeze(1))  # [B, N, d]
        t_src = fusion * ca_src + (1-fusion) * sa_src

        return t_src


class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = torch.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, query: Tensor,
                key: Tensor,
                value: Tensor,
                src_mask: Optional[Tensor] = None,
                ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = query
            # cross-attention
        if self.norm_first:
            x = x + torch.nan_to_num(self._sa_block(self.norm1(x), key, value, attn_mask=src_mask), 0)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + torch.nan_to_num(self._sa_block(x, key, value, attn_mask=src_mask), 0))
            x = self.norm2(x + self._ff_block(x))
        return x

    # self-attention block
    def _sa_block(self, query: Tensor,
                  key: Tensor,
                  value: Tensor,
                  attn_mask: Optional[Tensor] = None) -> Tensor:
        x = self.self_attn(query, key, value,
                           attn_mask=attn_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
