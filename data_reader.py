import numpy as np


class DataReader:

    @staticmethod
    def read(file: str) -> np.array:
        return np.loadtxt(usecols=(0, 1, 2), fname=file)
