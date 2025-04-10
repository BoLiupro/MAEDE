import torch
from torch import nn

from .positional_encoding import PositionalEncoding
from .transformer_layer import SpatialTransformerLayer, TemporalTransformerLayer
from .diffusion_encoding import DiffusionEncoding


class STFormer(nn.Module):
    """Spatial and temporal transformer network"""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.encoder_depth = config['encoder_depth']
        self.mlp_ratio = config['mlp_ratio']
        dropout = config['dropout']

        ## embedding
        self.input_embedding = nn.Linear(config['input_dim']*config['t_len'], config['embed_dim'])
        self.condition_embedding = nn.Linear(config['condition_dim'] * 4, config['embed_dim'])

        ## spatio-temporal positional encoding
        self.positional_encoding = PositionalEncoding(config['pos_dim'], config['pos_dim'], config['embed_dim'], dropout=dropout)

        ## diffusion parameters for encoding and decoding
        self.diffusion_encoding = DiffusionEncoding(config['num_steps'], config['embed_dim'], dropout=dropout)
        self.diffusion_decoding = DiffusionEncoding(config['num_steps'], config['embed_dim'], dropout=dropout)

        ## transformer encoders
        self.encoders = nn.ModuleList(
            [SpatialTransformerLayer(config['embed_dim'], config['mlp_ratio'], config['num_heads'], dropout=dropout)
             for _ in range(config['encoder_depth'])])

        # decoder
        # this might be helpful, from SimST
        self.node_embedding = nn.Parameter(torch.randn(config['num_nodes'], config['embed_dim']))
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer_feat = nn.Linear(config['embed_dim'], config['embed_dim'])
        self.output_layer = nn.Linear(config['embed_dim'], config['output_dim'])

    def encoding(self, input_data, condition, side_info, training=True):
        """Encoding process of TSFormer: patchify, positional encoding, mask, Transformer layers.

        Args:
            input_data (torch.Tensor): input with shape [B, N, L, D].
            condition: context nodes encoding with shape [B, N, L, D].
            side_info (Dict() torch.Tensor): side info dict.
        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """
        assert condition.shape[2] == 4

        batch_size, num_nodes, t_len, _ = input_data.shape
        condition_nodes = condition.shape[1]

        # embed input
        input_data = input_data.reshape(batch_size, num_nodes, -1)  # B, N, L*D
        encoder_input = torch.relu(self.input_embedding(input_data))            # B, N, D
        condition = condition.reshape(batch_size, condition_nodes, -1)  # B, N, L*D
        condition = torch.relu(self.condition_embedding(condition))            # B, N, D

        # generate parameters given diffusion step
        side_info['dpe'] = self.diffusion_encoding(side_info['diffusion_step'])  # B, emd_dim
        # side info encoding: spatial, temporal, external features, and diffusion step
        side_info['spe'], side_info['tpe'] = self.positional_encoding(side_info['spe'], side_info['tpe'])  # B, N, d / B, L, d

        # spatial transformer encoding
        encoding = encoder_input
        for layer in self.encoders:
            encoding = layer(encoding, condition, side_info, training)  # B, N, D
        return encoding

    def decoding(self, encoding, side_info):
        dpe = self.diffusion_decoding(side_info['diffusion_step']).unsqueeze(1)  # B, 1, D
        encoding = encoding + dpe + self.node_embedding[side_info['target_index']].unsqueeze(0)  # B, N, D
        prediction = self.output_dropout(torch.relu(self.output_layer_feat(encoding)))
        prediction = self.output_layer(prediction)  # B, N, D
        prediction = prediction.unsqueeze(2).transpose(2, 3)  # B, N, D, 1;  D = L here
        return prediction

    def forward(self, X, condition, side_info, training=True):
        """
        Args:
            X (torch.Tensor): ground truth signal with shape [B, N, L, d_x].
            side_info (Dict('spatial', 'external'(optional): torch.Tensor)): side info dict.
            training (bool): True in training stage and False in inference stage.
        Returns:
            reconstruction_full (torch.Tensor): reconstructed noise data with shape [B, N, L, d_x].
            mask_matrix (torch.Tensor): mask indicator with shape [B, N, L].
        """
        # encoding
        encoding = self.encoding(X, condition, side_info, training)
        # decoding
        prediction = self.decoding(encoding, side_info)
        return prediction


class STFormerForecasting(nn.Module):
    """Spatial and temporal transformer network"""

    def __init__(self, config):
        super().__init__()
        self.embed_dim = config['embed_dim']
        self.num_heads = config['num_heads']
        self.encoder_depth = config['encoder_depth']
        self.mlp_ratio = config['mlp_ratio']
        dropout = config['dropout']

        ## embedding
        self.input_embedding = nn.Linear(config['input_dim']*config['t_len'], config['embed_dim'])
        self.condition_embedding = nn.Linear(config['condition_dim'], config['embed_dim'])

        ## spatio-temporal positional encoding
        self.positional_encoding = PositionalEncoding(config['pos_dim'], config['pos_dim'], config['embed_dim'], dropout=dropout)

        ## diffusion parameters for encoding and decoding
        self.diffusion_encoding = DiffusionEncoding(config['num_steps'], config['embed_dim'], dropout=dropout)
        self.diffusion_decoding = DiffusionEncoding(config['num_steps'], config['embed_dim'], dropout=dropout)

        ## transformer encoders
        self.encoders = nn.ModuleList(
            [TemporalTransformerLayer(config['embed_dim'], config['mlp_ratio'], config['num_heads'], dropout=dropout)
             for _ in range(config['encoder_depth'])])

        # decoder
        # this might be helpful, from SimST
        #self.node_embedding = nn.Parameter(torch.randn(config['num_nodes'], config['embed_dim']))
        self.output_dropout = nn.Dropout(dropout)
        self.output_layer_feat = nn.Linear(config['embed_dim'], config['embed_dim'])
        self.output_layer = nn.Linear(config['embed_dim'], config['output_dim'])

    def encoding(self, input_data, condition, side_info, training=True):
        """Encoding process of TSFormer: patchify, positional encoding, mask, Transformer layers.

        Args:
            input_data (torch.Tensor): input with shape [B, N, L, D].
            condition: context nodes encoding with shape [B, N, L, D].
            side_info (Dict() torch.Tensor): side info dict.
        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """

        batch_size, num_nodes, t_len, _ = input_data.shape

        # embed input
        input_data = input_data.reshape(batch_size, num_nodes, -1)  # B, N, L*D
        encoder_input = torch.relu(self.input_embedding(input_data))            # B, N, L, d
        condition = torch.relu(self.condition_embedding(condition))            # B, N, L, d

        # generate parameters given diffusion step
        side_info['dpe'] = self.diffusion_encoding(side_info['diffusion_step'])  # B, emd_dim
        # side info encoding: spatial, temporal, external features, and diffusion step
        side_info['spe'], side_info['tpe'] = self.positional_encoding(side_info['spe'], side_info['tpe'])  # B, N, d / B, L, d

        # spatial transformer encoding
        encoding = encoder_input
        for layer in self.encoders:
            encoding = layer(encoding, condition, side_info, training)  # B, N, D
        return encoding

    def decoding(self, encoding, side_info):
        dpe = self.diffusion_decoding(side_info['diffusion_step']).unsqueeze(1)  # B, 1, D
        encoding = encoding + dpe #+ self.node_embedding.unsqueeze(0)  # B, N, D
        prediction = self.output_dropout(torch.relu(self.output_layer_feat(encoding)))
        prediction = self.output_layer(prediction)  # B, N, D
        prediction = prediction.unsqueeze(2).transpose(2, 3)  # B, N, D, 1;  D = L here
        return prediction

    def forward(self, X, condition, side_info, training=True):
        """
        Args:
            X (torch.Tensor): ground truth signal with shape [B, N, L, d_x].
            side_info (Dict('spatial', 'external'(optional): torch.Tensor)): side info dict.
            training (bool): True in training stage and False in inference stage.
        Returns:
            reconstruction_full (torch.Tensor): reconstructed noise data with shape [B, N, L, d_x].
            mask_matrix (torch.Tensor): mask indicator with shape [B, N, L].
        """
        # encoding
        encoding = self.encoding(X, condition, side_info, training)
        # decoding
        prediction = self.decoding(encoding, side_info)
        return prediction
