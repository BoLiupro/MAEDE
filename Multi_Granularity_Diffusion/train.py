import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
import os
import copy
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator
from Multi_Granularity_Diffusion.utils.multi_gran_generator import creat_coarse_data
from mgtsd_estimator import mgtsdEstimator
from trainer import Trainer
from pathlib import Path
import wandb
from Multi_Granularity_Diffusion.utils.utils import plot
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='mgtsd', help='model name')
    parser.add_argument('--cuda_num', type=str,
                        default='0', help='cuda number') #
    parser.add_argument('--result_path', type=str,
                        default='./results/', help='result path')
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=5e-05)
    parser.add_argument('--num_cells', type=int, default=320, help='number of cells in the rnn')
    parser.add_argument('--diff_steps', type=int,default=100, help='diff steps')
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--mg_dict', type=str, default='1_2_4',
                        help='the multi-granularity list, 1_4 means 1h and 4h, 1_4_8 means 1h, 4h and 8h')
    parser.add_argument('--num_gran', type=int, default=3,
                        help='the number of granularities, must be equal to the length of mg_dict')
    parser.add_argument('--share_ratio_list', type=str, default="1_0.6_0.4",
                        help='the share ratio list, 1_0.9, means that for the second granularity, 90% of the diffusion steps are shared with the finest granularity.')
    parser.add_argument('--weight_list', type=str, default="0.85_0.1_0.05",
                        help='the weight list, 0.9_0.1 means that the weight for the first granularity is 0.9 and the weight for the second granularity is 0.1.')
    
    parser.add_argument('--run_num', type=str, default="1",
                        help='the index of the run, used for the result file name')
    parser.add_argument('--wandb_space', type=str,
                        default="test", help='the space name of the wandb')
    parser.add_argument('--wandb_key', type=str, default="a813dcc064cfd0805c559d6f8b2e082191068896",
                        help='the key of the wandb, please replace it with your own key')
    parser.add_argument('--log_metrics', type=str2bool, default="True",
                        help='whether to log the metrics to the wandb when training. it will slow down the training process')

    # 返回一个命名空间，包含传递给命令的参数
    return parser.parse_args()

selected_nodes = [0,1,2,3,4]

def get_H(file_path):
    scaler = StandardScaler()
    graph_features_path = file_path # [T,dh,N,1]
    graph_features = np.load(graph_features_path,allow_pickle=True)  #
    graph_features = graph_features[:,:,selected_nodes,:]
    graph_features = graph_features.reshape(graph_features.shape[0], graph_features.shape[1], -1) # [T,F,N]
    graph_features = graph_features.reshape(graph_features.shape[0],-1) # [T,F]
    graph_features_std = scaler.fit_transform(graph_features) # 归一化
    return graph_features_std

def main():
    args = parse_args()
    model_name = args.model_name
    cuda_num = args.cuda_num
    result_path = args.result_path
    Path(result_path).mkdir(parents=True, exist_ok=True)
    epoch = args.epoch
    diff_steps = args.diff_steps
    num_gran = args.num_gran

    batch_size = args.batch_size
    mg_dict = [float(i) for i in str(args.mg_dict).split('_')]
    print(f"mg_dict:{mg_dict}")
    share_ratio_list = [float(i) for i in str(args.share_ratio_list).split('_')]
    weight_list = [float(i) for i in str(args.weight_list).split('_')]
    weights = weight_list
    print(f"share_ratio_list:{share_ratio_list}")
    learning_rate = args.learning_rate
    num_cells = args.num_cells
    if args.log_metrics:
        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_space, save_code=True, config=args)
    print(args)

    device = torch.device(
        f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu")

    print("================================================")
    print("prepare the dataset")

    # 总时间步现在只有4128了,这是由于H长度在USTD中被batchsize切割了
    T = 4128
    N = 5

    """multi-granularity condition"""
    h1 = get_H("")
    h2 = get_H("")

    # 加载原始时间序列数据
    data_path = ''
    data = np.load(data_path)  # [T, N]
    data = data + 1e-5 # epsilon 
    data = data.transpose(1,0)[:,:T] # [N,T]
    """先只跑5个点，点太多了服务器会崩"""
    data = data[selected_nodes,:]

    """选择研究点"""
    train_rate = 3000
    freq = "h"
    start = pd.Period("03-01-2024", freq=freq).to_timestamp()
    test_start = start + pd.Timedelta(train_rate,unit='h')
    train_data = data[:,:train_rate] # [N,T]
    test_data = data[:,train_rate:] # [N,T]
    print(train_data.shape) #[N,T] = [5,3072]
    print(test_data.shape) #[N,T] = [5,3072]


    train_dataset = ListDataset(
        [{"target": train_data[i,:], "start": start} for i in range(N)],
        freq=freq,
    )

    test_dataset = ListDataset(
        [{"target": test_data[i,:], "start": test_start} for i in range(N)],
        freq=freq,
    )

    train_grouper = MultivariateGrouper(max_target_dim=N) # max_target_dim = N
    test_grouper = MultivariateGrouper(max_target_dim=N) # max_target_dim = N
    from gluonts.dataset.field_names import FieldName

    dataset_train = list(train_grouper(train_dataset))
    dataset_test = list(test_grouper(test_dataset))

    for i, entry in enumerate(dataset_train):
        entry[FieldName.HIDDEN_STATE_1] = h1[:train_rate,:].tolist()
    for i, entry in enumerate(dataset_test):
        entry[FieldName.HIDDEN_STATE_1] = h1[train_rate:,:].tolist()
    for i, entry in enumerate(dataset_train):
        entry[FieldName.HIDDEN_STATE_2] = h2[:train_rate,:].tolist()
    for i, entry in enumerate(dataset_test):
        entry[FieldName.HIDDEN_STATE_2] = h2[train_rate:,:].tolist()

    data_train, data_test = creat_coarse_data(dataset_train=dataset_train,
                                                dataset_test=dataset_test,
                                                mg_dict=mg_dict)
    print("================================================")
    print("initlize the estimator")

    prediction_length = 3
    dh = 64
    estimator = mgtsdEstimator(
        target_dim=N, # N=15
        prediction_length=prediction_length,
        context_length=24,
        cell_type='LSTM',
        # input_size=N + num_gran*N + 4 + dh*N*3,  # 4*N+4 = 516
        input_size=4*N+4+dh*N, # N+3*N+4+dh*N=5+15+4+128*5=24+640=664   N*3+N+4+dh*N= 64+64*15=1024  4N+4+dh*N
        freq=freq,
        loss_type='l2',
        scaling=True,
        diff_steps=diff_steps,
        share_ratio_list=share_ratio_list,
        beta_end=0.1,
        beta_schedule="linear",
        weights=weights,
        num_cells=num_cells,
        num_gran=num_gran,
        trainer=Trainer(device=device,
                        epochs=epoch,
                        learning_rate=learning_rate,
                        num_batches_per_epoch=120,
                        batch_size=batch_size,
                        log_metrics=args.log_metrics,)
    )
    print("================================================")
    print("start training the network")
    predictor = estimator.train(data_train, num_workers=8, validation_data=data_test)
    # predictor = estimator.train(data_train, num_workers=8, validation_data=None)


    print("===============================================")
    print("make predictions")
    forecast_it, ts_it = make_evaluation_predictions(dataset=data_test,
                                                    predictor=predictor,
                                                    num_samples=100)
    forecasts = list(forecast_it)
    targets = list(ts_it)

    print("the number of days for targets")
    print(len(targets))
    print(forecasts[0].samples.shape)  
    print(targets[0].shape) 

    targets_list = []
    forecasts_list = []
    target_dim = estimator.target_dim
    target_columns = targets[0].iloc[:, :target_dim].columns
    for cur_gran_index, cur_gran in enumerate(mg_dict):
        targets_cur = []
        predict_cur = []
        predict_cur = copy.deepcopy(forecasts)

        for i in range(len(targets)):
            targets_cur.append(
                targets[i].iloc[:, (cur_gran_index * target_dim):((cur_gran_index + 1) * target_dim)])
            targets_cur[-1].columns = target_columns
        for day in range(len(forecasts)):
            predict_cur[day].samples = forecasts[day].samples[:, :,
                                                            (cur_gran_index * target_dim):((cur_gran_index + 1) * target_dim)]
            print(f'predict_cur:{predict_cur[day].samples.shape}')
        targets_list.append(targets_cur)
        forecasts_list.append(predict_cur)


    # Ignore all warnings
    warnings.filterwarnings("ignore")


    agg_metric_list = []
    for cur_gran_index, cur_gran in enumerate(mg_dict):
        evaluator = MultivariateEvaluator(quantiles=(np.arange(20) / 20.0)[1:],
                                        target_agg_funcs={'sum': np.sum})
        agg_metric, item_metrics = evaluator(targets_list[cur_gran_index], forecasts_list[cur_gran_index],
                                            num_series=len(data_test)/2)
        agg_metric_list.append(agg_metric)


    for cur_gran_index, cur_gran in enumerate(mg_dict):
        agg_metric = agg_metric_list[cur_gran_index]
        print(f"=======evaluation metrics for {cur_gran} h samples")
        
        print("CRPS:", agg_metric["mean_wQuantileLoss"])
        print("ND:", agg_metric["ND"])
        print("NRMSE:", agg_metric["NRMSE"])
        print("MSE",agg_metric["MSE"])
        print("CRPS-Sum:", agg_metric["m_sum_mean_wQuantileLoss"])
        print("ND-Sum:", agg_metric["m_sum_ND"])
        print("NRMSE-Sum:", agg_metric["m_sum_NRMSE"])

        CRPS = agg_metric["mean_wQuantileLoss"]
        ND = agg_metric["ND"]
        NRMSE = agg_metric["NRMSE"]
        CRPS_Sum = agg_metric["m_sum_mean_wQuantileLoss"]
        ND_Sum = agg_metric["m_sum_ND"]
        NRMSE_Sum = agg_metric["m_sum_NRMSE"]

        if args.log_metrics:
            wandb.log({f'CRPS_Sum_{cur_gran}': CRPS_Sum,
                    f'ND_Sum_{cur_gran}': ND_Sum, f'NRMSE_Sum_{cur_gran}': NRMSE_Sum})

        # test results for fine-grained dataset

        filename = f"{result_path}/output_{'Changsha'}_{model_name}_{mg_dict}h_{cur_gran}h_{diff_steps}_{weights}_ratio{share_ratio_list}.csv"
        if not os.path.exists(filename):
            with open(filename, mode="a") as f:
                f.write("epoch,model_name,CRPS,ND,NRMSE,CRPS_Sum,ND_Sum,NRMSE_Sum\n")

        result_str = f"{epoch}, {model_name}, {CRPS}, {ND}, {NRMSE}, {CRPS_Sum}, {ND_Sum}, {NRMSE_Sum}\n"
        with open(filename, mode="a") as f:  # append the column names to the file
            f.write(result_str)
        plot(targets_list[cur_gran_index][0], forecasts_list[cur_gran_index][0], prediction_length=prediction_length,
            fname=f"{result_path}/plot_{'Changsha'}_{model_name}_{mg_dict}h_{cur_gran}h_{diff_steps}_{weights}_ratio{share_ratio_list}.png")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()