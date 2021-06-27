from typing import Tuple

import numpy as np


class ConditionalGradient:

    def __init__(self, x0: np.array, data_set: np.array, tau: float):
        self.x_t = x0
        self.data_set = data_set

        self.list_of_tuples = data_set.tolist()

        self.tau = tau

        one_dimensional_data_set = []
        for entry in self.list_of_tuples:
            _, _, score = entry
            one_dimensional_data_set.append(score)
        one_dimensional_data_set = np.array(one_dimensional_data_set)
        vector_size = np.linalg.norm(one_dimensional_data_set)

        # The hessian is the 4-dimensional tensor we get by the cross product of the entries of E with themseles.
        # The smoothness factor of this hessian is the size of E when it is treated as a one dimensional vector.
        self.beta = vector_size

    def find_min_i_j(self, x_t: np.array) -> Tuple[int, int, float]:
        i_min = j_min = -1
        min_abs_val = np.infty
        min_sample_val = np.infty

        for i,j,val in self.list_of_tuples:
            abs_val = np.abs(x_t[int(i),int(j)] - val)

            if abs_val < min_abs_val:
                min_abs_val = abs_val
                i_min = int(i)
                j_min = int(j)
                min_sample_val = val

        return i_min, j_min, min_sample_val

    def solve_conditional_gradient_sub_problem(self, x_t: np.array, gradient: np.array) -> np.array:
        i_min, j_min, min_sample_val = self.find_min_i_j(x_t=x_t)

        v_t = np.zeros_like(x_t)

        v_t[i_min, j_min] = -np.sign(min_sample_val) * np.sqrt(self.tau)

        return v_t


    def gradient(self, x_t) -> np.array:
        gradient = np.zeros_like(x_t)

        for i, j, val in self.list_of_tuples:
            gradient[int(i), int(j)] = x_t[int(i), int(j)] - val

        return gradient

    def conditional_gradient_method(self, iterations_num: int) -> np.array:

        for t in range(1, iterations_num + 1):
            eta_t = 2 / (t + 1)

            gradient = self.gradient(x_t=self.x_t)

            v_t = self.solve_conditional_gradient_sub_problem(x_t=self.x_t, gradient=gradient)

            self.x_t = self.x_t - eta_t * (v_t - self.x_t)

        return self.x_t


