
import numpy as np


class FactorizedFormGradientDescent:
    def __init__(self, v0: np.array, u0: np.array, data_set: np.array, n: int, m: int, rank: int,
                 epsilon: float):
        self.v_t = v0
        self.u_t = u0
        self.t = 0
        self.n = n
        self.m = m
        self.data_set = data_set
        self.rank = rank

        self.list_of_tuples = data_set.tolist()

        self.epsilon = epsilon

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
            gradient[i, j] = x_t[i, j] - val

        return gradient

    def gradient_u(self, u_t: np.array, v_t: np.array) -> np.array:

        common_gradient_part = self.gradient_common(u_t @ np.transpose(v_t))
        return common_gradient_part @ v_t

    def gradient_v(self, u_t: np.array, v_t: np.array) -> np.array:

        common_gradient_part = self.gradient_common(u_t @ np.transpose(v_t))
        return u_t @ common_gradient_part

    def factorized_method_gradient_descent(self):

        gradient_u = self.gradient_u(u_t=self.u_t, v_t=self.v_t)
        gradient_v = self.gradient_v(u_t=self.u_t, v_t=self.v_t)

        joint_gradient = self.create_joint_gradient(gradient_u=gradient_u,
                                                    gradient_v=gradient_v)

        while np.linalg.norm(joint_gradient) > self.epsilon:
            gradient_u = self.gradient_u(u_t=self.u_t, v_t=self.v_t)
            gradient_v = self.gradient_v(u_t=self.u_t, v_t=self.v_t)

            joint_gradient = self.create_joint_gradient(gradient_u=gradient_u,
                                                        gradient_v=gradient_v)

            self.u_t = self.u_t - (1/self.beta) * gradient_u
            self.v_t = self.v_t - (1/self.beta) * gradient_v

        return self.u_t @ np.transpose(self.v_t)

    def create_joint_gradient(self, gradient_u: np.array, gradient_v: np.array) -> np.array:
        return np.array(gradient_u.tolist().extend(gradient_v.tolist()))
