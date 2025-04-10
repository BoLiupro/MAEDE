import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
from loguru import logger

"""
Define three main coordinate systems
    - WGS84: The coordinate system used by the GPS global satellite positioning system.
    - GCJ02: Mars coordinate system, the coordinate system formulated by the State Bureau of Surveying and Mapping of China.
    - BD09: The coordinate system used by Baidu Maps.
"""

# Access key for Baidu Maps
# Configuration viewable at: https://lbsyun.baidu.com/apiconsole/center#/home
ak = ''
lon_min=113.1767
lon_max=113.2967
lat_min=23.0711
lat_max=23.1911
lon_grids=25 
lat_grids=25
lon_step = (lon_max - lon_min) / lon_grids
lat_step = (lat_max - lat_min) / lat_grids

def get_grid_coordinates(grid_id):
    """
    根据网格编号计算网格的经纬度范围。
    """
    lat_index = (grid_id - 1) // lon_grids
    lon_index = (grid_id - 1) % lon_grids

    lon_start = lon_min + lon_index * lon_step
    lon_end = lon_start + lon_step
    lat_start = lat_min + lat_index * lat_step
    lat_end = lat_start + lat_step

    return lon_start, lon_end, lat_start, lat_end


# Convert latitude and longitude to Baidu map coordinates
def bd_latlng2xy(zoom, latitude, longitude):
    url = "https://api.map.baidu.com/geoconv/v1/"
    # For detailed parameters, refer to https://lbs.baidu.com/faq/api?title=webapi/guide/changeposition-base
    params = {
        "coords": str(longitude) + ',' + str(latitude),
        "from": "5",
        "to": "6",
        "ak": ak,
    }
    response = requests.get(url=url, params=params)
    result = response.json()
    logger.info(f'result: {result}')
    loc = result["result"][0]
    res = 2 ** (18 - zoom)  # Calculate the scaling factor
    x = loc['x'] / res
    y = loc['y'] / res
    return x, y


# Download map tiles
def download_tiles(city, zoom, latitude_start, latitude_stop, longitude_start, longitude_stop, satellite=True):
    # Create a save directory with a separate subdirectory for each city
    root_save = os.path.join("image", "Guangzhou")
    os.makedirs(root_save, exist_ok=True)

    # Perform coordinate conversion
    start_x, start_y = bd_latlng2xy(zoom, latitude_start, longitude_start)
    stop_x, stop_y = bd_latlng2xy(zoom, latitude_stop, longitude_stop)

    # Calculate tile range
    start_x = int(start_x // 256)
    start_y = int(start_y // 256)
    stop_x = int(stop_x // 256)
    stop_y = int(stop_y // 256)

    if start_x >= stop_x or start_y >= stop_y:
        logger.info("Invalid coordinates range")
        return

    logger.info(f'x range: {start_x} to {stop_x}')
    logger.info(f'y range: {start_y} to {stop_y}')

    # Loop to download each tile, using a thread pool of custom size, e.g., max_workers=666
    with ThreadPoolExecutor(max_workers=666) as executor:
        futures = []
        for x in range(start_x, stop_x):
            for y in range(start_y, stop_y):
                futures.append(executor.submit(download_tile, x, y, zoom, satellite, root_save))
        # Wait for all threads to complete
        for future in futures:
            future.result()


# Download an individual map tile
def download_tile(x, y, zoom, satellite, root_save):
    if satellite:
        # Satellite imagery URL
        url = f"http://shangetu0.map.bdimg.com/it/u=x={x};y={y};z={zoom};v=009;type=sate&fm=46&udt=20250108&app=webearth2&v=009&udt=20250108"
        filename = f"{zoom}_{x}_{y}_s.png"
    else:
        # Road map image URL
        url = f'http://online3.map.bdimg.com/tile/?qt=tile&x={x}&y={y}&z={zoom}&styles=pl&scaler=1&udt=20180810'
        filename = f"{zoom}_{x}_{y}_r.png"

    filename = os.path.join(root_save, filename)

    # Check if the file exists, download if it doesn't
    if not os.path.exists(filename):
        try:
            logger.info(f'downloading filename: {filename}')
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            logger.info(f"-- saving {filename}")
            with open(filename, 'wb') as f:
                f.write(response.content)
            time.sleep(random.random())  # Random sleep to reduce server load
        except requests.RequestException as e:
            logger.info(f"-- {filename} -> {e}")
    else:
        logger.info(f"File already exists: {filename}")


def download():
    # Define the latitude and longitude range of cities
    # Coordinate picker viewable at: https://api.map.baidu.com/lbsapi/getpoint/index.html
    grids = {}
    # """hot_nodes: [56, 63, 91, 84, 73]"""
    # """Node: 56"""
    Grids = [216, 217, 241, 265, 291, 292, 293, 316, 317, 341, 342, 366, 367, 368, 391, 392, 393, 416, 417, 418, 441, 442, 443, 467]
    """Node: 63"""
    Grids += [247, 272, 273, 296, 297, 298, 320, 321, 322, 323, 346, 347, 348, 371, 372, 396]
    """Node: 91"""
    Grids += [420, 421, 444, 445, 446, 447, 470, 471, 472, 493, 494, 495, 496, 497, 518, 519, 520, 521, 522, 544, 545]
    """Node: 84"""
    Grids += [373, 374, 398, 399, 400, 423, 424, 425, 449, 450, 474]
    """Node: 73"""
    Grids += [314, 315, 339, 364, 389, 390, 414, 415, 440, 464, 465, 466, 490, 491]
    """Node:100"""
    Grids += [475, 500]
    """Node:105"""
    Grids += [504, 505, 529]
    """Node:33"""
    Grids += [126, 127, 151]
    """Node:94"""
    Grids += [439]
    """Node:74"""
    Grids += [324, 349]
    """Node:125"""
    Grids += [542, 543, 549, 550, 566, 567, 568, 569, 570, 572, 573, 574, 591, 593]
    """Node:126"""
    Grids += [6, 7, 8, 535, 536, 32, 33, 560, 561, 562, 51, 52, 53, 57, 58, 585, 586, 76, 77, 78]
    """Node:127"""
    Grids += [16, 38, 39, 40, 64, 65, 66, 89, 90, 97, 114, 122, 139, 140, 147, 169, 170, 193, 194]
    """Node:120"""
    Grids += [592]
    """Node:78"""
    Grids += [345]
    # """Downstream task"""
    """广州"""
    # grids = {}
    # # hot_nodes: [11, 64, 6, 88, 23]
    # # Node: 11
    # Grids  = [20, 45, 46, 69, 70, 71, 94, 95, 96, 120, 121, 145, 146, 147, 169, 194, 219, 244]
    # # Node: 64
    # Grids += [245, 246, 269, 270, 271, 294, 295, 296, 319, 320, 321, 344, 345, 368, 369]
    # # Node: 6
    # Grids += [11, 34, 35, 36, 60, 61, 84, 85, 86, 110, 111, 112, 135, 136, 137, 159, 160]
    # # Node: 88
    # Grids += [355, 379, 380, 381, 382, 404, 405, 406, 407, 430, 431, 432, 455, 456, 457, 481, 482]
    # # Node: 23
    # Grids += [66, 90, 91, 92, 93, 115, 116, 117, 140, 141, 164, 165, 166, 190]
    # # Cold nodes: [141, 142, 143, 144, 82]
    # # Node: 141
    # Grids +=[612]
    # # Node: 142
    # Grids += [514, 515, 516, 517, 521, 522, 523, 524, 525, 539, 541, 542, 546, 547, 548, 549, 550]
    # # Node: 143
    # Grids += [9, 10, 526, 527, 528, 529, 530, 26, 551, 552, 553, 554, 557, 558, 51, 52, 54, 576]
    # # Node: 144
    # Grids += [15, 16, 17, 41, 88]
    # # Node: 82
    # Grids += [332]
    # # Normal nodes: [111, 94, 137, 22, 129]
    # # Node: 111
    # Grids += [450]
    # # Node: 94
    # Grids += [385]
    # # Node: 137
    # Grids += [590, 615]
    # # Node: 22
    # Grids += [62, 87]
    # # Node: 129
    # Grids += [555, 556, 581]
    for i in Grids:
        lon_start, lon_end, lat_start, lat_end = get_grid_coordinates(i)
        # 字典中加入,key为网格编号，value为经纬度范围
        grids[i] = (lat_start, lat_end, lon_start, lon_end)

    # zoom = 16  # Coarse zoom level
    zoom = 19  # 19 for Fine zoom level
    satellite = True  # Satellite image (if False, download road images)

    # Loop through the cities and download the corresponding satellite images
    for city, coordinates in grids.items():
        logger.info(f"Downloading tiles for {city}...")
        lat_start, lat_stop, lon_start, lon_stop = coordinates
        download_tiles(city, zoom, lat_start, lat_stop, lon_start, lon_stop, satellite)


# 读取图片特征
feats = pd.read_csv("image_features.csv")

# 获取某个地理区域的所有图片特征
def GeoFuse(zoom, latitude_start, latitude_stop, longitude_start, longitude_stop):
    start_x, start_y = bd_latlng2xy(zoom, latitude_start, longitude_start)
    stop_x, stop_y = bd_latlng2xy(zoom, latitude_stop, longitude_stop)

    start_x, start_y = int(start_x // 256), int(start_y // 256)
    stop_x, stop_y = int(stop_x // 256), int(stop_y // 256)

    if start_x >= stop_x or start_y >= stop_y:
        logger.info("Invalid coordinates range")
        return []

    logger.info(f'x range: {start_x} to {stop_x}')
    logger.info(f'y range: {start_y} to {stop_y}')

    current_grid_feats = []
    for x in range(start_x, stop_x):
        for y in range(start_y, stop_y):
            df = feats[feats['file_name'] == f'19_{x}_{y}_s.png']['image_features']
            if not df.empty:
                current_grid_feats.extend(df.tolist())  # 确保是 list 格式

    return current_grid_feats


def fuse():
    """Changsha"""
    nodes = {
        56: [216, 217, 241, 265, 291, 292, 293, 316, 317, 341, 342, 366, 367, 368, 391, 392, 393, 416, 417, 418, 441,442, 443, 467],
        63: [247, 272, 273, 296, 297, 298, 320, 321, 322, 323, 346, 347, 348, 371, 372, 396],
        91: [420, 421, 444, 445, 446, 447, 470, 471, 472, 493, 494, 495, 496, 497, 518, 519, 520, 521, 522, 544, 545],
        84: [373, 374, 398, 399, 400, 423, 424, 425, 449, 450, 474],
        73: [314, 315, 339, 364, 389, 390, 414, 415, 440, 464, 465, 466, 490, 491],  # hot
        100:[475, 500],
        105:[504, 505, 529],
        33:[126, 127, 151],
        94:[439],
        74:[324,349], # normal
        125:[516, 517, 525, 542, 543, 549, 550, 566, 567, 568, 569, 570, 572, 573, 574, 591, 593],
        126:[6, 7, 8, 535, 536, 32, 33, 560, 561, 562, 51, 52, 53, 57, 58, 585, 586, 76, 77, 78],
        127:[16, 38, 39, 40, 64, 65, 66, 89, 90, 97, 114, 122, 139, 140, 147, 169, 170, 193, 194],
        120:[592],
        78:[345]
    }
    # """Guangzhou"""
    # nodes = {
    # 11 : [20, 45, 46, 69, 70, 71, 94, 95, 96, 120, 121, 145, 146, 147, 169, 194, 219, 244],
    # 64 : [245, 246, 269, 270, 271, 294, 295, 296, 319, 320, 321, 344, 345, 368, 369],
    # 6 : [11, 34, 35, 36, 60, 61, 84, 85, 86, 110, 111, 112, 135, 136, 137, 159, 160],
    # 88 : [355, 379, 380, 381, 382, 404, 405, 406, 407, 430, 431, 432, 455, 456, 457, 481, 482],
    # 23 : [66, 90, 91, 92, 93, 115, 116, 117, 140, 141, 164, 165, 166, 190],
    # 141 : [612],
    # 142 : [514, 515, 516, 517, 521, 522, 523, 524, 525, 539, 541, 542, 546, 547, 548, 549, 550],
    # 143 : [9, 10, 526, 527, 528, 529, 530, 26, 551, 552, 553, 554, 557, 558, 51, 52, 54, 576],
    # 144 : [15, 16, 17, 41, 88],
    # 82 : [332],
    # 111 : [450],
    # 94 : [385],
    # 137 : [590, 615],
    # 22 : [62, 87],
    # 129 : [555, 556, 581]
    # }
    zoom = 19  # Fine zoom level

    node_feats = {}

    for node, grids in nodes.items():
        current_node_feats = []
        for grid in grids:
            lon_start, lon_end, lat_start, lat_end = get_grid_coordinates(grid)
            grid_feats = GeoFuse(zoom, lat_start, lat_end, lon_start, lon_end)
            current_node_feats.extend(grid_feats)  # **使用 extend 而不是 append**

        node_feats[node] = current_node_feats  # **正确存储到字典**

    # 将 node_feats 转换为 DataFrame 并保存 CSV
    df = pd.DataFrame.from_dict(node_feats, orient='index')
    df.index.name = 'Node'
    df.to_csv("node_features.csv")

    print("Features saved to node_features.csv")



if __name__ == "__main__":
    fuse()
    # download()
