import os

import numpy as np

import partition_data
from conditional_gradient_method import ConditionalGradient
from data_reader import DataReader
from factorized_form_gradient_descent import FactorizedFormGradientDescent
from rank_projection import RankProjectionAlg


def main():
    curr_directory = os.path.dirname(os.path.realpath(__file__))
    ml_100k_data = os.path.join(curr_directory, 'ml-100k', 'u.data')

    data = DataReader.read(ml_100k_data)

    training_set, test_set = partition_data.split(data_set=data, test_fraction=1 / 3)

    max_m = -np.infty
    max_n = -np.infty

    for i, j, _ in training_set:
        if i > max_m:
            max_m = i
        if j > max_n:
            max_n = j

    x0 = np.zeros(max_m, max_n)
    rank = 10
    tau = rank
    u0 = np.zeros(max_m, rank)
    v0 = np.zeros(max_n, rank)

    rank_projection_solver = RankProjectionAlg(x0=x0, rank=rank, data_set=training_set)
    factorized_solver = FactorizedFormGradientDescent(u0=u0, v0=v0, rank=rank, data_set=training_set)
    conditional_gradient_solver = ConditionalGradient(x0=x0, data_set=training_set, tau=tau)

    iterations_num = 1000

    rank_projection_solution = rank_projection_solver.solve_rank_projection(iterations_num=iterations_num)
    factorized_solution = factorized_solver.factorized_method_gradient_descent(iterations_num=iterations_num)
    conditional_gradient_solution = conditional_gradient_solver.conditional_gradient_method(
        iterations_num=iterations_num)


if __name__ == "__main__":
    # execute only if run as a script
    main()
