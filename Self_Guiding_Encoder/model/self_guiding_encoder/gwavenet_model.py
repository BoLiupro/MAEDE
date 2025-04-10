from models import init_net, BaseModel
import torch
from .gwavenet import GWaveNetEncoder, GWaveNetDecoder

class GWaveNetModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        # modify options for the model
        parser.add_argument('--delete_col', default=['u_speed', 'v_speed', 'latitude', 'longitude'], help='')
        parser.add_argument('--use_adj', default=True)
        return parser

    def __init__(self, opt, model_config):
        super().__init__(opt, model_config)
        """
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['mae']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['Encoder', 'Decoder']

        # specify metrics you want to evaluate the model. The training/test scripts will call functions in order:
        # <BaseModel.compute_metrics> compute metrics for current batch
        # <BaseModel.get_current_metrics> compute and return mean of metrics, clear evaluation cache for next evaluation
        self.metric_names = ['MAE']

        # define networks. The model variable name should begin with 'self.net'
        model_config['input_dim'] = opt.y_dim + opt.covariate_dim
        model_config['output_dim'] = opt.t_len
        self.netEncoder = GWaveNetEncoder(model_config)
        self.netEncoder = init_net(self.netEncoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.netDecoder = GWaveNetDecoder(model_config)
        self.netDecoder = init_net(self.netDecoder, opt.init_type, opt.init_gain, opt.gpu_ids)
        self.H = []
        # define loss functions
        if self.isTrain:
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer = torch.optim.Adam(list(self.netEncoder.parameters())
                                               +list(self.netDecoder.parameters()), lr=opt.lr, betas=(0.9, 0.999))
            self.optimizers.append(self.optimizer)



    def set_input(self, input):
        """
        parse input for one epoch. data should be stored as self.xxx which would be adopted in self.forward().
        we construct the spatial embedding vectors here.
        :param input: dict
        :return: None
        """
        # network inputs
        self.pred_gt = input['pred'].to(self.device)  # [batch, num_n, time, d_x]  这里设置了历史时间步
        if 'feat' in input.keys():
            self.covariate = input['feat'].to(self.device)  # [batch, num_n, time, d_c]
        self.missing_mask = input['missing_mask'].to(self.device)
        # side info dic contains embedding vectors for attributes (e.g., spatial, temporal, covariates, diffusion steps)
        self.adj = [(input['adj'][0] / input['adj'][0].sum(dim=-1, keepdim=True)) .to(self.device),
                    (input['adj'][0].t() / input['adj'][0].t().sum(dim=-1, keepdim=True)).to(self.device)]  # same in the batch


    def forward(self, training=True):
        if hasattr(self, 'covariate'):
            inputs = torch.cat([self.pred_gt, self.covariate], dim=-1)
        else:
            inputs = self.pred_gt
        self.mean, self.node_mask = self.netEncoder(inputs, self.adj, training)
        sample = self.mean
        self.prediction,hidden_state = self.netDecoder(sample, self.adj)
        self.H.extend(hidden_state)
        # print(len(self.H))

    def backward(self):
        self.loss_mae = self.mae_loss(self.prediction, self.pred_gt, self.node_mask)
        loss = self.loss_mae
        loss.backward()

    def mae_loss(self, pred, gt, node_mask=None):
        if node_mask is None:
            node_mask = torch.ones_like(gt).to(self.device)
        if node_mask.shape != gt.shape:
            node_mask = node_mask.unsqueeze(-1)
        loss = torch.sum(torch.abs(pred - gt) * node_mask) / node_mask.sum()
        return loss

    def mae(self, pred, gt, node_mask=None):
        if node_mask is None:
            node_mask = torch.ones_like(gt).to(self.device)
        if node_mask.shape != gt.shape:
            node_mask = node_mask.unsqueeze(-1)
        pred = pred * self.opt.scale + self.opt.mean
        gt = gt * self.opt.scale + self.opt.mean
        return torch.sum(torch.abs(pred - gt) * node_mask) / node_mask.sum()

    def cache_results(self):
        loss = self.mae(self.prediction, self.pred_gt)
        self._add_to_cache('pred', self.prediction, reverse_norm=True)
        self._add_to_cache('gt', self.pred_gt, reverse_norm=True)
        self._add_to_cache('MAE', loss.unsqueeze(0))

    def compute_metrics(self):
        self.metric_MAE = self.results['MAE'].mean()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netEncoder, self.netDecoder], True)
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def get_conditional_representation(self):
        return self.H
