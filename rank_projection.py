import numpy as np


class RankProjectionAlg:

    def __init__(self, x0: np.array, data_set: np.array, rank: int):
        self.x_t = x0
        self.data_set = data_set
        self.rank = rank

        self.list_of_tuples = data_set.tolist()

        one_dimensional_data_set = []
        for entry in self.list_of_tuples:
            _, _, score = entry
            one_dimensional_data_set.append(score)
        one_dimensional_data_set = np.array(one_dimensional_data_set)
        vector_size = np.linalg.norm(one_dimensional_data_set)

        # The hessian is the 4-dimensional tensor we get by the cross product of the entries of E with themseles.
        # The smoothness factor of this hessian is the size of E when it is treated as a one dimensional vector.
        self.beta = vector_size

    def gradient(self, x_t) -> np.array:
        gradient = np.zeros_like(x_t)

        for i, j, val in self.list_of_tuples:
            gradient[i, j] = x_t[i, j] - val

        return gradient

    def project(self, vec: np.array) -> np.array:
        u, s, vh = np.linalg.svd(vec)

        # TODO validate that this is non negative and does what is expected
        s_list = s.tolist()
        s_list[self.rank + 1:] = 0.0

        s = np.array(s)

        return (u * s) @ vh

    def solve_rank_projection(self, iterations_num: int):

        for _ in range(iterations_num):
            gradient = self.gradient(self.x_t)
            self.x_t = self.project(x_t - (1/self.beta) * gradient)

        return x_t
