import os

import numpy as np

import partition_data
from conditional_gradient_method import ConditionalGradient
from data_reader import DataReader
from factorized_form_gradient_descent import FactorizedFormGradientDescent
from rank_projection import RankProjectionAlg


def evaluate_score(data: np.array, low_rank_approximation: np.array) -> float:
    running_sum = 0.0
    for i, j, val in data:
        running_sum += np.power(low_rank_approximation[int(i), int(j)] - val, 2)
    return running_sum


def main():
    curr_directory = os.path.dirname(os.path.realpath(__file__))
    ml_100k_data = os.path.join(curr_directory, 'ml-100k', 'u.data')

    data = DataReader.read(ml_100k_data)

    training_set, test_set = partition_data.split(data_set=data, test_fraction=1 / 3)

    test_methods_on_data_set(data, test_set, training_set)

    ml_1m_data = os.path.join(curr_directory, 'ml-1m', 'ratings.dat')

    data = DataReader.read(ml_1m_data, delimiter='::')

    training_set, test_set = partition_data.split(data_set=data, test_fraction=1 / 3)

    test_methods_on_data_set(data, test_set, training_set)


def test_methods_on_data_set(data, test_set, training_set):
    max_m = -np.infty
    max_n = -np.infty
    for i, j, _ in data:
        if i > max_m:
            max_m = int(i)
        if j > max_n:
            max_n = int(j)
    x0 = np.zeros((max_m + 1, max_n + 1))
    rank = 10
    tau = rank
    u0 = np.zeros((max_m + 1, rank))
    v0 = np.zeros((max_n + 1, rank))
    rank_projection_solver = RankProjectionAlg(x0=x0, rank=rank, data_set=training_set)
    factorized_solver = FactorizedFormGradientDescent(u0=u0, v0=v0, rank=rank, data_set=training_set)
    conditional_gradient_solver = ConditionalGradient(x0=x0, data_set=training_set, tau=tau)
    iterations_num = 10
    print(f'Starting rank projection solution.\n')
    try:
        rank_projection_solution = rank_projection_solver.solve_rank_projection(iterations_num=iterations_num)
    except Exception as ex:
        rank_projection_solution = None
        print(f'rank_projection_solver failed with exception: \n{ex}')
    print(f'Finished rank projection solution.\n')
    print(f'Starting factorized solution.\n')
    factorized_solution = factorized_solver.factorized_method_gradient_descent(iterations_num=iterations_num)
    print(f'Finished factorized solution.\n')
    print(f'Starting conditional gradient solution.\n')
    conditional_gradient_solution = conditional_gradient_solver.conditional_gradient_method(
        iterations_num=iterations_num)
    print(f'Finished conditional gradient solution.\n')
    if rank_projection_solution is not None:
        rank_projection_result = evaluate_score(data=test_set, low_rank_approximation=rank_projection_solution)
    else:
        rank_projection_result = np.inf
    factorization_result = evaluate_score(data=test_set, low_rank_approximation=factorized_solution)
    conditional_gradient_result = evaluate_score(data=test_set, low_rank_approximation=conditional_gradient_solution)
    print(f'rank_projection_result = {rank_projection_result}\n')
    print(f'factorization_result = {factorization_result}\n')
    print(f'conditional_gradient_result = {conditional_gradient_result}\n')


if __name__ == "__main__":
    # execute only if run as a script
    main()
