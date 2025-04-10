# implemented by p0werHu
from .Citybase_dataset import CityDataset


class GuangzhouDataset(CityDataset):
    """
    Note that the beijing air quality dataset contains a lot of missing values, we need to handle this explicitly.
    """

    def __init__(self, opt):
        """
        load data give options
        """
        dist_path = '' # Adjacency matrix
        data_path = '' # Traffic time series
        test_nodes_path = ''

        super().__init__(opt, dist_path, data_path, test_nodes_path)
