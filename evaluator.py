import numpy as np


class Evaluator:
    def __init__(self, test_set, iteration_num):
        self.test_set = test_set
        self.iteration_num = iteration_num
        self.scores = np.zeros(self.iteration_num)

    def evaluate(self, curr_iter, curr_xt):
        running_sum = 0.0
        for i, j, val in self.test_set:
            running_sum += np.power(curr_xt[int(i), int(j)] - val, 2)

        self.scores[curr_iter] = running_sum