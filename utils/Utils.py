import math

from ..utils import MyParser

def get_params():
    args = MyParser.get_parser()
    lon_min = args.lon_min
    lon_max = args.lon_max
    lat_min = args.lat_min
    lat_max = args.lat_max
    lon_grids = args.lon_grids
    lat_grids = args.lat_grids
    lon_step = (lon_max - lon_min) / lon_grids
    lat_step = (lat_max - lat_min) / lat_grids
    return args,lon_min, lon_max, lat_min, lat_max, lon_grids, lat_grids, lon_step, lat_step

# 计算网格编号
def get_grid_index(lon, lat):
    args,lon_min, lon_max, lat_min, lat_max, lon_grids, lat_grids, lon_step, lat_step = get_params()
    if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
        lon_index = math.floor((lon - lon_min) / lon_step)
        lat_index = math.floor((lat - lat_min) / lat_step)
        return lat_index * lon_grids + lon_index + 1
    return None

def get_grid_coordinates(grid_id):
    args,lon_min, lon_max, lat_min, lat_max, lon_grids, lat_grids, lon_step, lat_step = get_params()
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

def get_grid_center(grid_id):
    """
    根据网格编号计算网格的中心经纬度。
    """
    lon_start, lon_end, lat_start, lat_end = get_grid_coordinates(grid_id)
    lon_center = (lon_start + lon_end) / 2
    lat_center = (lat_start + lat_end) / 2
    return (lon_center, lat_center)

def compute_distance(girdIndex_1, gridIndex_2):
    """
    计算两个网格之间的距离。
    """
    lon1, lat1 = get_grid_center(girdIndex_1)
    lon2, lat2 = get_grid_center(gridIndex_2)
    # 使用欧氏距离计算
    return math.sqrt((lon1 - lon2) ** 2 + (lat1 - lat2) ** 2)


# 计算两个经纬度点之间的距离（Haversine公式）
def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371.0  # 地球半径（单位：km）
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c