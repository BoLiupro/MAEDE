"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
import os


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)
    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

#####################################
# evaluation metrics
#####################################

import numpy as np

#####################################
# evaluation metrics
#####################################

def _mse_with_missing(y, label, missing_mask):
    """
    Args:
        y: np.array [..., D]
        label: np.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        mse: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    valid_count = np.sum(valid_mask)

    mse = (((y - label) ** 2) * valid_mask).sum() / (valid_count + 1e-7)
    return mse


def _rmse_with_missing(y, label, missing_mask):
    """
    Args:
        y: np.array [..., D]
        label: np.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        rmse: float
    """
    mse = _mse_with_missing(y, label, missing_mask)
    return np.sqrt(mse)


def _nrmse_with_missing(y, label, missing_mask):
    """
    Args:
        y: np.array [..., D]
        label: np.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        nrmse: float
    """
    rmse = _rmse_with_missing(y, label, missing_mask)
    range_val = np.max(label) - np.min(label)
    return rmse / (range_val + 1e-6)


def _r2_with_missing(y, label, missing_mask):
    """
    Args:
        y: np.array [..., D]
        label: np.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        r2: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    valid_count = np.sum(valid_mask)

    mean_label = (label * valid_mask).sum() / valid_count
    ss_total = (((label - mean_label) ** 2) * valid_mask).sum()
    ss_residual = (((y - label) ** 2) * valid_mask).sum()
    
    return 1 - (ss_residual / (ss_total + 1e-7))

def _nd_with_missing(y, label, missing_mask):
    """
    Normalized Deviation (ND) - 计算时跳过分母为 0 的样本
    Args:
        y: np.array [..., D] - 预测值
        label: np.array [..., D] - 真实值
        missing_mask: [..., 1] or [...] - 1 表示缺失，0 表示有效
    Returns:
        nd: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask

    abs_error = np.abs(y - label) * valid_mask  # 计算绝对误差
    abs_label = np.abs(label) * valid_mask      # 计算真实值的绝对值

    # 计算每个样本的分母（避免除 0）
    denominator = abs_label.sum(axis=-1)  # 在最后一个维度上求和
    valid_samples = denominator > 0       # 仅选择分母不为 0 的样本

    if np.sum(valid_samples) == 0:
        return 0  # 如果所有样本的分母都是 0，则返回 0

    nd = abs_error.sum(axis=-1)[valid_samples] / denominator[valid_samples]  # 仅计算有效样本
    return np.mean(nd)  # 计算有效样本的平均 ND


def _ndsum_with_missing(y, label, missing_mask):
    """
    Summed Normalized Deviation (NDsum) - 计算时跳过分母为 0 的样本
    Args:
        y: np.array [..., D] - 预测值
        label: np.array [..., D] - 真实值
        missing_mask: [..., 1] or [...] - 1 表示缺失，0 表示有效
    Returns:
        ndsum: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask

    abs_error = np.abs(y - label) * valid_mask  # 计算绝对误差
    abs_label = np.abs(label) * valid_mask      # 计算真实值的绝对值

    # 计算每个样本的分母
    denominator = abs_label.sum(axis=-1)  # 在最后一个维度上求和
    valid_samples = denominator > 0       # 仅选择分母不为 0 的样本

    if np.sum(valid_samples) == 0:
        return 0  # 如果所有样本的分母都是 0，则返回 0

    ndsum = abs_error.sum(axis=-1)[valid_samples] / denominator[valid_samples]  # 仅计算有效样本
    return np.sum(ndsum)  # 计算有效样本的 NDsum（累加）



def _nrmse_sum_with_missing(y, label, missing_mask):
    """
    Summed Normalized Root Mean Square Error (NRMSEsum)
    Args:
        y: np.array [..., D]
        label: np.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        nrmse_sum: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask

    mse = (((y - label) ** 2) * valid_mask).sum()
    mean_label = (label * valid_mask).sum() / np.sum(valid_mask)
    rmse_sum = np.sqrt(mse / (np.sum(valid_mask) + 1e-7))

    return rmse_sum / (mean_label + 1e-7)


def _crps_sum_with_missing(y, label, missing_mask):
    """
    Cumulative Ranked Probability Score (CRPS) Sum
    Args:
        y: np.array [time, num_sample, num_m, dy]
        label: np.array [time, num_m, dy]
        missing_mask: [time, num_m, 1] or [time, num_m]
    Returns:
        crps_sum: float
    """
    y = y.transpose(1, 0, 2, 3)  # [num_sample, time, num_m, dy]
    
    def quantile_loss(target, forecast, q: float, eval_points) -> float:
        return 2 * np.sum(
            np.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
        )

    quantiles = np.arange(0.05, 1.0, 0.05)
    crps_list = []
    
    for q in quantiles:
        crps = quantile_loss(label, np.quantile(y, q, axis=0), q, (1 - missing_mask))
        crps_list.append(crps)

    return np.mean(crps_list)


def _rmse_with_missing(y, label, missing_mask):
    """
    Args:
        y: nd.array [..., D]
        label: nd.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        rmse: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    valid_count = np.sum(valid_mask)

    rmse = np.sqrt((((y - label) ** 2) * valid_mask).sum() / (valid_count + 1e-7))

    return rmse


def _mae_with_missing(y, label, missing_mask):
    """
    Args:
        y: nd.array [..., D]
        label: nd.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        mae: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    valid_count = np.sum(valid_mask)

    mae = np.abs((y-label) * valid_mask).sum() / valid_count
    return mae

def _mape_with_missing(y, label, missing_mask):
    """
    Args:
        y: nd.array [..., D]
        label: nd.array [..., D]
        missing_mask: [..., 1] or [...]
    Returns:
        mape: float
    """
    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape == y.shape[:-1]:
        missing_mask = missing_mask[..., np.newaxis]
    valid_mask = 1 - missing_mask
    valid_mask = valid_mask * (np.abs(label) > 0.0001)
    valid_count = np.sum(valid_mask)

    mape = np.abs((y-label) / (label+1e-6) * valid_mask).sum() / valid_count
    return mape

def _quantile_CRPS_with_missing(y, label, missing_mask):
    """
    Args:
        y: nd.array [time, num_sample, num_m, dy]
        label: nd.array [time, num_m, dy]
        missing_index: [time, num_m, 1] or [time, num_m]
    Returns:
        CRPS: float
    """
    y = y.transpose(1, 0, 2, 3) # [num_sample, time, num_m, dy]
    def quantile_loss(target, forecast, q: float, eval_points) -> float:
        return 2 * np.sum(
            np.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
        )

    def calc_denominator(label, valid_mask):
        return np.sum(np.abs(label * valid_mask))

    if len(missing_mask.shape) != len(label.shape) and missing_mask.shape[:2] == y.shape[:2]:
        missing_mask = missing_mask[:, :, np.newaxis]

    valid_mask = 1 - missing_mask
    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(label, valid_mask)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = np.quantile(y, quantiles[i], axis=0)
        q_loss = quantile_loss(label, q_pred, quantiles[i], valid_mask)
        CRPS += q_loss / denom
    return CRPS / len(quantiles)
