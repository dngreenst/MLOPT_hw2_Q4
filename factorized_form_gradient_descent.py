
import numpy as np


class FactorizedFormGradientDescent:
    def __init__(self, v0: np.array, u0: np.array, data_set: np.array, rank: int):
        self.v_t = v0
        self.u_t = u0
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

    def gradient_common(self, x_t) -> np.array:
        gradient = np.zeros_like(x_t)

        for i, j, val in self.list_of_tuples:
            gradient[int(i), int(j)] = x_t[int(i), int(j)] - val

        return gradient

    def gradient_u(self, u_t: np.array, v_t: np.array) -> np.array:

        common_gradient_part = self.gradient_common(u_t @ np.transpose(v_t))
        return common_gradient_part @ v_t

    def gradient_v(self, u_t: np.array, v_t: np.array) -> np.array:

        common_gradient_part = self.gradient_common(u_t @ np.transpose(v_t))
        return np.transpose(common_gradient_part) @ u_t

    def factorized_method_gradient_descent(self, iterations_num: int):

        for _ in range(iterations_num):
            gradient_u = self.gradient_u(u_t=self.u_t, v_t=self.v_t)
            gradient_v = self.gradient_v(u_t=self.u_t, v_t=self.v_t)

            self.u_t = self.u_t - (1/self.beta) * gradient_u
            self.v_t = self.v_t - (1/self.beta) * gradient_v

        return self.u_t @ np.transpose(self.v_t)

