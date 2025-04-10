import pandas as pd
import requests
from loguru import logger
from utils import Utils

# Access key for Baidu Maps
ak = ''  # Baidu
args, lon_min, lon_max, lat_min, lat_max, lon_grids, lat_grids, lon_step, lat_step = Utils.get_params()

feats = pd.read_csv("image_features.csv")

# Convert latitude and longitude to Baidu map coordinates
def bd_latlng2xy(zoom, latitude, longitude):
    url = "https://api.map.baidu.com/geoconv/v1/"
    params = {
        "coords": f"{longitude},{latitude}",
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


def main():
    nodes = {
        56: [216, 217, 241, 265, 291, 292, 293, 316, 317, 341, 342, 366, 367, 368, 391, 392, 393, 416, 417, 418, 441,
             442, 443, 467],
        63: [247, 272, 273, 296, 297, 298, 320, 321, 322, 323, 346, 347, 348, 371, 372, 396],
        91: [420, 421, 444, 445, 446, 447, 470, 471, 472, 493, 494, 495, 496, 497, 518, 519, 520, 521, 522, 544, 545],
        84: [373, 374, 398, 399, 400, 423, 424, 425, 449, 450, 474],
        73: [314, 315, 339, 364, 389, 390, 414, 415, 440, 464, 465, 466, 490, 491],
    }
    zoom = 19  # Fine zoom level

    node_feats = {}

    for node, grids in nodes.items():
        current_node_feats = []
        for grid in grids:
            lon_start, lon_end, lat_start, lat_end = Utils.get_grid_coordinates(grid)
            grid_feats = GeoFuse(zoom, lat_start, lat_end, lon_start, lon_end)
            current_node_feats.extend(grid_feats)  # **使用 extend 而不是 append**

        node_feats[node] = current_node_feats  # **正确存储到字典**

    # 将 node_feats 转换为 DataFrame 并保存 CSV
    df = pd.DataFrame.from_dict(node_feats, orient='index')
    df.index.name = 'Node'
    df.to_csv("node_features.csv")

    print("Features saved to node_features.csv")


if __name__ == "__main__":
    main()
