import numpy as np


class DataReader:

    @staticmethod
    def read(file: str, delimiter = None) -> np.array:
        # return np.loadtxt(usecols=(0, 1, 2), fname=file, delimiter=delimiter)
        kwargs = {'delimiter': delimiter, 'usecols': (0,1,2)}
        return DataReader.numpy_loadtxt_memory_friendly(file, 1000000, usecols=(0,1,2), delimiter=delimiter)

    @staticmethod
    def numpy_loadtxt_memory_friendly(the_file, max_bytes = 1000000, **loadtxt_kwargs):
        numpy_arrs = []
        with open(the_file, 'rb') as f:
            i = 0
            while True:
                print(i)
                some_lines = f.readlines(max_bytes)
                if len(some_lines) == 0:
                    break
                vec = np.loadtxt(some_lines, **loadtxt_kwargs)
                if len(vec.shape) < 2:
                    vec = vec.reshape(1,-1)
                numpy_arrs.append(vec)
                i+=len(some_lines)
        return np.concatenate(numpy_arrs, axis=0)
