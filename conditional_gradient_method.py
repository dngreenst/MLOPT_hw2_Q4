from typing import Tuple

import numpy as np

from evaluator import Evaluator


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

        for i, j, val in self.list_of_tuples:
            diff_val = x_t[int(i), int(j)] - val
            abs_val = np.abs(x_t[int(i), int(j)] - val)

            if abs_val < min_abs_val:
                min_abs_val = abs_val
                i_min = int(i)
                j_min = int(j)
                min_sample_val = diff_val

        return i_min, j_min, min_sample_val

    def get_max_eigenvector(self, A: np.array) -> np.array:
        w = np.random.normal(size=A.shape[1])
        T = 20

        for _ in range(T):
            new_W_prime = A @ w
            w = (new_W_prime) / np.linalg.norm(new_W_prime)

        return w

    def get_max_singular_value(self, A: np.array) -> float:
        max_eigenvector = self.get_max_eigenvector(A.T @ A)
        return np.sqrt(np.linalg.norm(A.T @ A @ max_eigenvector))

    def solve_conditional_gradient_sub_problem(self, x_t: np.array, gradient: np.array) -> np.array:
        # get maximal eigenvalue of gradient
        max_eigenvalue = self.get_max_singular_value(gradient)
        # u, s, vh = np.linalg.svd(gradient, full_matrices=False)
        # # w, _ = np.linalg.eig(gradient)
        # max_eigenvalue = max(s)
        epsilon = 0.1

        # build matrix A
        m, n = gradient.shape

        A = np.zeros((n + m, n + m))

        A[0:m, 0:m] = max_eigenvalue * (1 + epsilon) * np.eye(m)
        A[0:m, m:m + n] = -gradient
        A[m:m + n, 0:m] = -np.transpose(gradient)
        A[m:n + m, m:n + m] = max_eigenvalue * (1 + epsilon) * np.eye(n)

        # solve w^T A w
        # w, P = np.linalg.eigh(A)
        max_eigenvector = self.get_max_eigenvector(A)
        # max_eigenvector2 = P[:,-1]

        # u* = w[1:m]/||w[1:m]
        u = max_eigenvector[0:m] / np.linalg.norm(max_eigenvector[0:m])
        # v* = w[m+1:]/||w[m+1:]
        v = max_eigenvector[m:] / np.linalg.norm(max_eigenvector[m:])

        v_t = self.tau * np.outer(u, v)

        return v_t

    def gradient(self, x_t) -> np.array:
        gradient = np.zeros_like(x_t)

        for i, j, val in self.list_of_tuples:
            gradient[int(i), int(j)] = x_t[int(i), int(j)] - val

        return gradient

    def conditional_gradient_method(self, iterations_num: int, evaluator: Evaluator) -> np.array:

        for t in range(1, iterations_num + 1):
            eta_t = 2 / (t + 1)

            gradient = self.gradient(x_t=self.x_t)

            v_t = self.solve_conditional_gradient_sub_problem(x_t=self.x_t, gradient=gradient)

            self.x_t = self.x_t + eta_t * (v_t - self.x_t)

            evaluator.evaluate(t - 1, self.x_t)

        return self.x_t

    def evaluate_score(self) -> float:
        running_sum = 0.0
        for i, j, val in self.list_of_tuples:
            running_sum += np.power(self.x_t[int(i), int(j)] - val, 2)
        return running_sum
