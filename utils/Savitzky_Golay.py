import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from utils import Utils

args,lon_min, lon_max, lat_min, lat_max, lon_grids, lat_grids, lon_step, lat_step = Utils.get_params()

filename = ''
data = pd.read_csv(os.path.join(args.traffic_flow_filepath,args.city, filename))

# Extract the specific column (replace 73 with the index you need)
node_num = 73
context_length = 72
column_data = data.iloc[:context_length, node_num]

# Apply Savitzky-Golay filter
# You can adjust the window_length and polyorder
window_length = 11  # Must be an odd number
polyorder = 3  # Polynomial order

savgol_filtered = savgol_filter(column_data, window_length=window_length, polyorder=polyorder)
# Ensure no negative values
savgol_filtered = np.maximum(savgol_filtered, 0)

# Plot the original and filtered data
plt.figure(figsize=(12, 6))
plt.plot(column_data, marker='o', linestyle='-', markersize=3, alpha=0.5, label='Original Data')
plt.plot(savgol_filtered, color='green', linewidth=2, label=f'Savitzky-Golay Filter (window={window_length}, polyorder={polyorder})')
plt.title(f'Savitzky-Golay Filter with Non-Negative Values,windowlength{window_length}_polyorder{polyorder}')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
