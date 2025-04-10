import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='')
    city = 'Guangzhou'
    parser.add_argument('--year', type=int, default=2024,choices=[2022,2024], help='year of the data')
    parser.add_argument('--data_filepath', type=str, default='',help='Path to data directory')
    parser.add_argument('--map_filepath', type=str, default='', help='Path to data directory')
    parser.add_argument('--clustered_points_filepath', type=str, default='', help='Path to cluster points file')
    parser.add_argument('--staying_points_filepath', type=str, default='', help='Path to staying points file')
    parser.add_argument('--original_data_filepath', type=str, default='', help='Path to original data file')
    parser.add_argument('--grid_node_map_filepath', type=str, default='', help='Path to grid node map file')
    parser.add_argument('--traffic_flow_filepath', type=str, default='', help='Path to traffic flow file')
    parser.add_argument('--adj_filepath', type=str, default='', help='Path to adjacency matrix file')
    parser.add_argument('--scope', type=int, default=1, help='Scope value')
    parser.add_argument('--time_step', type=int, default=60, help='')
    if city == 'Changsha':
        """Changsha"""
        parser.add_argument('--city', type=str, default='Changsha',choices=['Guangzhou','Changsha'], help='City name')
        parser.add_argument('--lon_min', type=float, default=112.915, help='Minimum longitude')
        parser.add_argument('--lon_max', type=float, default=113.035, help='Maximum longitude')
        parser.add_argument('--lat_min', type=float, default=28.115, help='Minimum latitude')
        parser.add_argument('--lat_max', type=float, default=28.235, help='Maximum latitude')
        parser.add_argument('--lon_grids', type=int, default=25, help='Number of longitude grids')
        parser.add_argument('--lat_grids', type=int, default=25, help='Number of latitude grids')
        parser.add_argument('--num_nodes', type=int, default=128, help='')
        parser.add_argument('--unconnected_node_num', type=int, default=3, help='')
    elif city == 'Guangzhou':
        """Gaungzhou"""
        parser.add_argument('--city', type=str, default='Guangzhou',choices=['Guangzhou','Changsha'], help='City name')
        parser.add_argument('--lon_min', type=float, default=113.1767, help='Minimum longitude')
        parser.add_argument('--lon_max', type=float, default=113.2967, help='Maximum longitude')
        parser.add_argument('--lat_min', type=float, default=23.0711, help='Minimum latitude')
        parser.add_argument('--lat_max', type=float, default=23.1911, help='Maximum latitude')
        parser.add_argument('--lon_grids', type=int, default=25, help='Number of longitude grids')
        parser.add_argument('--lat_grids', type=int, default=25, help='Number of latitude grids')
        parser.add_argument('--num_nodes', type=int, default=145, help='')
        parser.add_argument('--unconnected_node_num', type=int, default=4, help='')

    # args = parser.parse_args()
    # 处理未知参数
    args, unknown = parser.parse_known_args()
    return args
