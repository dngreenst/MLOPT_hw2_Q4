
import numpy as np


def split(data_set: np.array, test_fraction):
    max_data_index = int(np.ceil(len(data_set) * (1 - test_fraction)))

    data = np.array(data_set.tolist()[:max_data_index])
    test = np.array(data_set.tolist()[max_data_index:])

    return data, test
