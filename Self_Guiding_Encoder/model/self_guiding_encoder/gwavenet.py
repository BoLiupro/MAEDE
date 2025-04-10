import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask import MaskGenerator
import random
# adopted from https://github.com/nnzhan/Graph-WaveNet/blob/master/model.py


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class GWaveNetEncoder(nn.Module):
    def __init__(self,
                 model_config,
                 kernel_size=2,
                 blocks=2,
                 layers=3):
        super(GWaveNetEncoder, self).__init__()
        self.dropout = model_config['dropout']
        self.blocks = blocks
        self.layers = layers
        self.mask_ratio = model_config['mask_ratio']

        in_dim = model_config['input_dim']
        residual_channels = model_config['embed_dim']
        dilation_channels = model_config['embed_dim']
        end_channels = model_config['end_dim']
        skip_channels = 4 * model_config['embed_dim']

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.mask_generator = MaskGenerator()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.empty_token = nn.Parameter(torch.randn([in_dim]))

        self.supports_len = 2
        output_temporal_length = 4
        output_temporal_length -= 1
        receptive_field = 1 + output_temporal_length

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation += 1
                receptive_field += additional_scope
                additional_scope += kernel_size - 1
                self.gconv.append(gcn(dilation_channels,residual_channels,self.dropout,support_len=self.supports_len))

        self.receptive_field = receptive_field

        self.end_conv = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

    def forward(self, input, adj, mask_node=True):
        if self.mask_ratio > 0 and mask_node:
            mask = self.mask_generator(input, self.mask_ratio) # B, N, L
            _mask = mask.unsqueeze(-1) # B, N, L, 1
            input = input * (1 - _mask) + self.empty_token * _mask
        else:
            mask = None

        input = input.permute([0, 3, 1, 2])
        """h_hist!!!!!"""
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]
            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip                
            x = self.gconv[i](x, adj)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        x = F.relu(skip)
        x = self.end_conv(x).permute([0, 2, 3, 1]) # [64,128,4,64]
        """
        The final shape of x is [B, N, L, D], where:
            B = Batch size
            N = Number of nodes
            L = Number of time steps (or spatial dependencies)
            D = Feature dimension (typically related to the embedding size of the model)
        """
        """提取conditional representaion H!!"""
        return x, mask

class GWaveNetDecoder(nn.Module):
    def __init__(self,
                 model_config,
                 kernel_size=2,
                 layers=3):
        super(GWaveNetDecoder, self).__init__()
        self.dropout = model_config['dropout']
        self.layers = layers

        in_dim = model_config['end_dim']
        residual_channels = model_config['embed_dim']
        dilation_channels = model_config['embed_dim']
        end_channels = 4 * model_config['embed_dim']
        skip_channels = 4 * model_config['embed_dim']
        end_channels_128 = end_channels
        end_channels_64 = 64
        out_dim = model_config['output_dim']

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.H = []

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))

        receptive_field = 1
        self.supports_len = 2

        for i in range(layers):
            # dilated convolutions
            self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                               out_channels=dilation_channels,
                                               kernel_size=(1,kernel_size),dilation=1))

            self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                             out_channels=dilation_channels,
                                             kernel_size=(1, kernel_size), dilation=1))

            # 1x1 convolution for residual connection
            self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=residual_channels,
                                                 kernel_size=(1, 1)))

            # 1x1 convolution for skip connection
            self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                             out_channels=skip_channels,
                                             kernel_size=(1, 1)))
            self.bn.append(nn.BatchNorm2d(residual_channels))
            receptive_field += kernel_size - 1
            self.gconv.append(gcn(dilation_channels,residual_channels,self.dropout,support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels_128,
                                    out_channels=end_channels_64,
                                    kernel_size=(1,1),
                                    bias=True)
        self.end_conv_3 = nn.Conv2d(in_channels=end_channels_64,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

    def forward(self, input, adj):
        input = input.permute([0, 3, 1, 2])
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        # WaveNet layers
        for i in range(self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]
            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            x = self.gconv[i](x, adj)
            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        # x = F.relu(skip)
        # H = x # [B,dh,N,1]
        # x = F.relu(self.end_conv_1(x))
        # x = self.end_conv_2(x)
        # x = x.permute([0, 2, 1, 3]) 
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = F.relu(self.end_conv_2(x))
        H = x # [B,dh,N,1]
        x = self.end_conv_3(x)
        x = x.permute([0, 2, 1, 3]) 
        """
        The final shape of x is [B, N, D, L], where:
            B = Batch size
            N = Number of nodes
            L = Number of time steps (or spatial dependencies)
            D = Feature dimension (typically related to the embedding size of the model)
        """
        """提取conditional representaion H!!"""
        # [B,N,n_pred,D] = [B,N,t_len,1] d_x = 1
        return x,H

#################
# previous version
#################

# class GWaveNetEncoder(nn.Module):
#     def __init__(self,
#                 model_config):
#         super(GWaveNetEncoder, self).__init__()
#
#         in_dim = model_config['input_dim']
#         embed_dim = model_config['embed_dim']
#         kernel_size = model_config['kernel_size']
#         layers = model_config['num_layers']
#         dropout = model_config['dropout']
#         self.dropout = dropout
#         self.layers = layers
#         self.mask_ratio = model_config['mask_ratio']
#
#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.residual_convs = nn.ModuleList()
#         # self.skip_convs = nn.ModuleList()
#         self.bn = nn.ModuleList()
#         self.gconv = nn.ModuleList()
#         self.pooling = nn.ModuleList()
#         self.start_conv = nn.Conv2d(in_channels=in_dim,
#                                     out_channels=embed_dim,
#                                     kernel_size=(1,1))
#         self.mask_generator = MaskGenerator()
#         self.empty_token = nn.Parameter(torch.randn([in_dim]))
#
#         # self.num_nodes = 207
#         # self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, embed_dim))
#         # self.nodevec2 = nn.Parameter(torch.randn(embed_dim, self.num_nodes))
#
#         adj_len = 2
#         receptive_field = 1
#         dilation = 1
#         padding_size = []
#         additional_scope = kernel_size - 1
#         for i in range(layers):
#             padding_size.append((kernel_size - 1) * dilation)
#
#             # dilated convolutions
#             self.filter_convs.append(nn.Conv2d(in_channels=embed_dim,
#                                                out_channels=embed_dim,
#                                                kernel_size=(1,kernel_size),dilation=dilation))
#             self.gate_convs.append(nn.Conv2d(in_channels=embed_dim,
#                                              out_channels=embed_dim,
#                                              kernel_size=(1, kernel_size), dilation=dilation))
#             # 1x1 convolution for residual connection
#             self.residual_convs.append(nn.Conv2d(in_channels=embed_dim,
#                                                  out_channels=embed_dim,
#                                                  kernel_size=(1, 1)))
#             # 1x1 convolution for skip connection
#             # self.skip_convs.append(nn.Conv2d(in_channels=embed_dim,
#             #                                      out_channels=embed_dim,
#             #                                      kernel_size=(1, 1)))
#             self.bn.append(nn.BatchNorm2d(embed_dim))
#             self.gconv.append(gcn(embed_dim, embed_dim, dropout, support_len=adj_len))
#             receptive_field += additional_scope
#
#         for i in range(layers-1):
#             self.pooling.append(nn.MaxPool2d((1, 2)))
#
#         self.end_conv_mean = nn.Conv2d(in_channels=embed_dim,
#                                     out_channels=embed_dim,
#                                     kernel_size=(1, 1),
#                                     bias=True)
#         self.end_conv_var = nn.Conv2d(in_channels=embed_dim,
#                                   out_channels=embed_dim,
#                                   kernel_size=(1,1),
#                                   bias=True)
#
#         self.padding_size = padding_size
#
#     def forward(self, input, adj, mask_node=True):
#         """
#         Args:
#             input: B, N, L, D
#             adj: N, N
#         Returns:
#         """
#         if self.mask_ratio > 0 and mask_node:
#             mask = self.mask_generator(input, self.mask_ratio) # B, N, L
#             _mask = mask.unsqueeze(-1) # B, N, L, 1
#             input = input * (1 - _mask) + self.empty_token * _mask
#         else:
#             mask = None
#         skip = None
#
#         # adp_adj = torch.eye(self.num_nodes, self.num_nodes).to(input.device) + F.softmax(
#         #     F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
#         # adj.append(adp_adj)
#
#         x = self.start_conv(input.permute([0, 3, 1, 2]))  # B, D, N, L
#         # WaveNet layers
#         for i in range(self.layers):
#
#             #            |----------------------------------------|     *residual*
#             #            |                                        |
#             #            |    |-- conv -- tanh --|                |
#             # -> dilate -|----|                  * ------- 1x1 -- + -->	*input*
#             #                 |-- conv -- sigm --|
#
#             residual = x
#
#             # padding
#             residual_padding = nn.functional.pad(residual, (self.padding_size[i], 0, 0, 0))
#             # dilated convolution
#             filter = self.filter_convs[i](residual_padding)
#             filter = torch.tanh(filter)
#             gate = self.gate_convs[i](residual_padding)
#             gate = torch.sigmoid(gate)
#             x = filter * gate
#             # if skip is None:
#             #     skip = self.skip_convs[i](x)
#             # else:
#             #     skip = skip[:, :, :, -x.size(3):] + self.skip_convs[i](x)
#
#             x = self.gconv[i](x, adj)
#             x = x + self.residual_convs[i](residual)
#             x = self.bn[i](x)
#             x = F.relu(x)
#             if i != self.layers-1:
#                 x = self.pooling[i](x)
#
#         x_mean = self.end_conv_mean(x).permute([0, 2, 3, 1])
#         x_std = self.end_conv_var(x).permute([0, 2, 3, 1])
#         x_std = 0.1 + 0.9 * F.softplus(x_std)
#         x_var = x_std ** 2
#         return x_mean, x_var, mask
#
#
# class GWaveNetDecoder(nn.Module):
#     def __init__(self,
#                 model_config):
#         super(GWaveNetDecoder, self).__init__()
#
#         embed_dim = model_config['embed_dim']
#         kernel_size = model_config['kernel_size']
#         layers = model_config['num_layers']
#         dropout = model_config['dropout']
#         self.dropout = dropout
#         self.layers = layers
#
#         self.filter_convs = nn.ModuleList()
#         self.gate_convs = nn.ModuleList()
#         self.residual_convs = nn.ModuleList()
#         # self.skip_convs = nn.ModuleList()
#         self.bn = nn.ModuleList()
#         self.gconv = nn.ModuleList()
#         self.pooling = nn.ModuleList()
#
#         # self.num_nodes = 207
#         # self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, embed_dim))
#         # self.nodevec2 = nn.Parameter(torch.randn(embed_dim, self.num_nodes))
#
#         adj_len = 2
#         receptive_field = 1
#         padding_size = []
#         additional_scope = kernel_size - 1
#         dilation = 1
#         for i in range(layers):
#             padding_size.append((kernel_size - 1) * dilation)
#
#             # dilated convolutions
#             self.filter_convs.append(nn.Conv2d(in_channels=embed_dim,
#                                                out_channels=embed_dim,
#                                                kernel_size=(1,kernel_size),dilation=dilation))
#             self.gate_convs.append(nn.Conv2d(in_channels=embed_dim,
#                                              out_channels=embed_dim,
#                                              kernel_size=(1, kernel_size), dilation=dilation))
#             # 1x1 convolution for residual connection
#             self.residual_convs.append(nn.Conv2d(in_channels=embed_dim,
#                                                  out_channels=embed_dim,
#                                                  kernel_size=(1, 1)))
#             # 1x1 convolution for skip connection
#             # self.skip_convs.append(nn.Conv2d(in_channels=embed_dim,
#             #                                      out_channels=embed_dim,
#             #                                      kernel_size=(1, 1)))
#             self.bn.append(nn.BatchNorm2d(embed_dim))
#             self.gconv.append(gcn(embed_dim, embed_dim, dropout, support_len=adj_len))
#             receptive_field += additional_scope
#
#         for i in range(layers-1):
#             self.pooling.append(nn.Upsample(scale_factor=(1, 2), mode='nearest'))
#
#         self.end_conv_1 = nn.Conv2d(in_channels=embed_dim,
#                                   out_channels=model_config['output_dim'],
#                                   kernel_size=(1,1),
#                                   bias=True)
#         self.padding_size = padding_size
#
#     def forward(self, input, adj):
#         """
#         Args:
#             input: B, N, L, D
#             adj: N, N
#         Returns:
#         """
#
#         x = input.permute([0, 3, 1, 2])  # B, D, N, L
#         skip = None
#
#         # adp_adj = torch.eye(self.num_nodes, self.num_nodes).to(input.device) + F.softmax(
#         #     F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
#         # adj.append(adp_adj)
#
#         # WaveNet layers
#         for i in range(self.layers):
#
#             #            |----------------------------------------|     *residual*
#             #            |                                        |
#             #            |    |-- conv -- tanh --|                |
#             # -> dilate -|----|                  * ------- 1x1 -- + -->	*input*
#             #                 |-- conv -- sigm --|
#
#             residual = x
#
#             # padding
#             residual_padding = nn.functional.pad(residual, (self.padding_size[i], 0, 0, 0))
#             # dilated convolution
#             filter = self.filter_convs[i](residual_padding)
#             filter = torch.tanh(filter)
#             gate = self.gate_convs[i](residual_padding)
#             gate = torch.sigmoid(gate)
#             x = filter * gate
#
#             # if skip is None:
#             #     skip = self.skip_convs[i](x)
#
#             x = self.gconv[i](x, adj)
#             x = x + self.residual_convs[i](residual)
#             x = self.bn[i](x)
#             x = F.relu(x)
#             if i != self.layers-1:
#                 x = self.pooling[i](x)
#
#         x = self.end_conv_1(x)
#         x = x.permute([0, 2, 3, 1])  # B, N, L, D
#         return x
