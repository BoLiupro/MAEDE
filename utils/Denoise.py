import os

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from utils import Utils

args,lon_min, lon_max, lat_min, lat_max, lon_grids, lat_grids, lon_step, lat_step = Utils.get_params()

def savitzky_golay_denoise(filename, window_length=11, polyorder=3):
    data = pd.read_csv(os.path.join(args.traffic_flow_filepath,args.city, filename),header=None)
    # 遍历数据集的每一列，对每一列进行滤波
    for i in range(data.shape[1]):
        column_data = data.iloc[:, i]
        savgol_filtered = savgol_filter(column_data, window_length=window_length, polyorder=polyorder)
        savgol_filtered = np.maximum(savgol_filtered, 0)
        data.iloc[:, i] = savgol_filtered
    return data

if __name__ == '__main__':
    filename = ''
    window_length = 11
    polyorder = 3
    data = savitzky_golay_denoise(filename, window_length, polyorder)

    print(data.isnull().sum())

    data.to_csv(os.path.join(args.traffic_flow_filepath,args.city, 'denoised_' + filename), index=False)
    # save as numpy
    np.save(os.path.join(args.traffic_flow_filepath, args.city,'denoised_' + filename[:-4] + '.npy'), data.values)
