# implemented by p0werHu
import datetime

from data.base_dataset import BaseDataset
import numpy as np
import pandas as pd
from scipy import sparse

class CityDataset(BaseDataset):
    """
    Note that the beijing air quality dataset contains a lot of missing values, we need to handle this explicitly.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        parser.set_defaults(y_dim=1, covariate_dim=2+64, spatial_dim=67)
        # parser.set_defaults(y_dim=129, covariate_dim=2, spatial_dim=64)
        return parser

    def __init__(self, opt, dist_path, data_path, test_nodes_path):
        super().__init__(opt)
        """
        load data give options
        """
        self.opt = opt
        self.time_division = {
            'train': [0.0, 0.95],
            'val': [0.95, 0.96],
            'test': [0, 1.0]
        }

        self.raw_data = self.load_feature(data_path, self.time_division[opt.phase])
        self.A = self.load_adj(dist_path)

        # divide data into train, val, test
        ## get division index
        self.opt.__dict__.update({'num_nodes': self.A.shape[0]})
        self.test_node_index = self.get_node_division(test_nodes_path, num_nodes=self.raw_data['pred'].shape[0])
        self.train_node_index = np.setdiff1d(np.arange(self.raw_data['pred'].shape[0]), self.test_node_index)

        # data format check
        self._data_format_check()

    
    def average_by_granularity(self, data, gran):
        """
        对数据按粒度 gran 进行平均处理
        :param data: 原始数据，形状为 [T, N]
        :param gran: 需要计算的粒度
        :return: 处理后的数据，形状仍为 [T, N]
        """
        T, N = data.shape
        new_data = np.zeros((T, N))
        
        i = 0
        while i < T:
            end = min(i + gran, T)  # 确保窗口不超过数据边界
            avg_value = np.mean(data[i:end], axis=0)  # 计算窗口均值
            new_data[i:end, :] = avg_value  # 赋值给窗口内所有元素
            i += gran  # 移动窗口
        
        return new_data


    def load_feature(self, data_path, time_division, add_time_in_day=True, add_day_in_week=True,add_graph_feature=True):
        # 1️⃣ Load data from NumPy file
        X = pd.read_csv(data_path,header=None)  # Load the NumPy file
        X = X.values
        gran = 2
        num_samples, num_nodes = X.shape  # [T, N]
        X = self.average_by_granularity(X,gran)

        # 2️⃣ Expand dimensions to match the shape [N, T, 1]
        X = np.expand_dims(X, axis=-1).transpose((1, 0, 2))  # From [T, N] to [N, T, 1]


        # 3️⃣ Add time-based features
        feature_list = []

        if add_time_in_day:
            # Generate time-of-day features
            time_ind = np.linspace(0, 1, num=num_samples, endpoint=False)  # Normalize time-of-day to [0, 1]
            time_in_day = np.tile(time_ind, [num_nodes, 1]).transpose((1, 0))  # [T, N]
            time_in_day = np.expand_dims(time_in_day, axis=-1)  # [T, N, 1]
            feature_list.append(time_in_day)

        if add_day_in_week:
            # Generate day-of-week features
            dow = np.arange(num_samples) % 7  # Generate day-of-week values (0-6)
            dow_tiled = np.tile(dow, [num_nodes, 1]).transpose((1, 0))  # [T, N]
            dow_tiled = np.expand_dims(dow_tiled, axis=-1)  # [T, N, 1]
            feature_list.append(dow_tiled)

        """图特征添加！"""
        if add_graph_feature:
            # graph_feat = np.load("") 
            # graph_feat = np.random.randn(num_nodes, 1024)
            graph_feat = pd.read_csv("").to_numpy()
            graph_feat = np.tile(graph_feat, [num_samples, 1, 1]) # [T,N,F]
            graph_feat.transpose(1,0,2) # [N,T,F]
            feature_list.append(graph_feat)

        # 4️⃣ Concatenate features along the last axis
        if feature_list:
            feat = np.concatenate(feature_list, axis=-1).transpose((1, 0, 2))  # [N, T, F]
        else:
            feat = np.zeros((num_nodes, num_samples, 0))  # No additional features

        # 5️⃣ Handle missing values
        missing_index = np.zeros(X.shape)
        missing_index[X == 0] = 1

        # 6️⃣ Generate time list (assuming 5-minute intervals)
        # 设置起始时间为 2024-06-01 00:00:00
        start_time = datetime.datetime.strptime('2024-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        # 生成时间列表，每个时间步的间隔是1小时，共2060个时间步
        time_list = [np.datetime64(start_time + t * datetime.timedelta(hours=1)) for t in range(num_samples)]
        # 转换为NumPy数组
        time_list = np.array(time_list)
        # 转换为Unix时间戳（秒为单位）
        time_list = ((time_list - np.datetime64('1970-01-01T00:00:00')) / np.timedelta64(1, 's')).astype(np.int64)

        # 7️⃣ Data normalization
        self.add_norm_info(np.mean(X), np.std(X))
        X = (X - self.opt.mean) / self.opt.scale

        # 8️⃣ Data division
        data_length = X.shape[1]
        start_index, end_index = int(time_division[0] * data_length), int(time_division[1] * data_length)
        X = X[:, start_index:end_index]
        missing_index = missing_index[:, start_index:end_index]
        feat = feat[:, start_index:end_index]
        time_list = time_list[start_index:end_index]

        # 9️⃣ Return final data dictionary
        data = {
            'time': time_list,
            'pred': X,
            'missing': missing_index,
            'feat': feat
        }
        return data

    def load_adj(self, npz_filename):
        adj_mx = self.load_npz(npz_filename)
        return adj_mx
    
    def load_npz(self, npz_file):
        """
        Load a sparse adjacency matrix from an npz file and return a NumPy array.
        """
        try:
            # Load the sparse matrix from the npz file
            adj_mx = sparse.load_npz(npz_file)
            # Convert the sparse matrix to a dense NumPy array
            adj_mx = adj_mx.toarray()
        except Exception as e:
            print('Unable to load data from ', npz_file, ':', e)
            raise
        return adj_mx
