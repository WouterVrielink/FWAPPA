import math
import numpy as np
from numba import jit


def two_dim_bench_functions():
    """
    First term are bounds, second term is the correction required to set the benchmark to 0.
    """
    return {
        easom: [((-100, 100), (-100, 100)), -1],
        branin: [((-5, 15), (-5, 15)), 0.39788735772973816],
        goldstein_price: [((-2, 2), (-2, 2)), 3],
        martin_gaddy: [((-20, 20), (-20, 20)), 0],
        six_hump_camel: [((-3, 3), (-2, 2)), -1.031628453489877]
    }


def n_dim_bench_functions():
    return {
        sphere: (-100, 100),
        tablet: (-100, 100),
        cigar: (-100, 100),
        elipse: (-100, 100),
        ackley: (-100, 100),
        schwefel: (-500, 500),
        rastrigrin: (-5.12, 5.12),
        rosenbrock: (-5, 10),
        griewank: (-600, 600)
    }


def two_dim_non_centered_bench_functions():
    return {
        easom: [(math.pi, math.pi)],
        branin: [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)],
        goldstein_price: [(0, -1)],
        six_hump_camel: [(0.0898, -0.7126), (-0.0898, 0.7126)],
        martin_gaddy: [(5, 5)]
    }


def n_dim_non_centered_bench_functions():
    return {
        schwefel: [(420.9687)],
        rosenbrock: [(1)]
    }


def param_shift(params, value):
    if isinstance(value, tuple):
        return [param + value for param, value in zip(params, value)]
    return [param + value for param in params]


def apply_add(bench_function, domain, value=10, name='_add'):
    def new_fun(params):
        params = param_shift(params, value)

        return bench_function(params)

    new_fun.__name__ = bench_function.__name__ + name + (str(value) if name == '_add' else '')

    if isinstance(value, tuple):
        return new_fun, [(min - value, max - value) for (min, max), value in zip(domain, value)]
    return new_fun, [(min - value, max - value) for (min, max) in domain]


def six_hump_camel(params):
    first_term = (4 - 2.1 * (params[0] ** 2) + (params[0] ** 4) / 3) * params[0] ** 2
    second_term = params[0] * params[1]
    third_term = (-4 + 4 * (params[1] ** 2)) * params[1] ** 2

    return first_term + second_term + third_term


def martin_gaddy(params):
    first_term = (params[0] - params[1]) ** 2
    second_term = ((params[0] + params[1] - 10) / 3) ** 2

    return first_term + second_term


def goldstein_price(params):
    first_term = 1 + ((params[0] + params[1] + 1) ** 2) * (19 - 14 * params[0] + 3 * params[0] ** 2 - 14 * params[1] + 6 * params[0] * params[1] + 3 * params[1] ** 2)
    second_term = 30 + ((2 * params[0] - 3 * params[1]) ** 2) * (18 - 32 * params[0] + 12 * params[0] ** 2 + 48 * params[1] - 36 * params[0] * params[1] + 27 * params[1] ** 2)

    return first_term * second_term


def branin(params):
    first_term = params[1] - (5.1 / (4 * math.pi ** 2)) * params[0] ** 2 + (5 / math.pi) * params[0] - 6
    second_term = 10 * (1 - 1 / (8 * math.pi)) * math.cos(params[0])

    return first_term ** 2 + second_term + 10


def easom(params):
    return -math.cos(params[0]) * math.cos(params[1]) * math.exp(-(params[0] - math.pi) ** 2 - (params[1] - math.pi) ** 2)


def rosenbrock(params):
    return sum([100 * (params[i + 1] - params[i] ** 2) ** 2 + (params[i] - 1) ** 2 for i in range(len(params) - 1)])


def ackley(params):
    first_term = -20 * math.exp(-0.2 * math.sqrt((1 / len(params)) * sum([param ** 2 for param in params])))
    second_term = math.exp((1 / len(params)) * sum([math.cos(2 * math.pi * param) for param in params]))

    return first_term - second_term + 20 + math.e


def griewank(params):
    first_term = sum([(param ** 2) / 4000 for param in params])
    second_term = np.prod([math.cos(param / math.sqrt(i + 1)) for i, param in enumerate(params)])

    return 1 + first_term + second_term


def rastrigrin(params):
    return 10 * len(params) + sum([param ** 2 - 10 * math.cos(2 * math.pi * param) for param in params])


def schwefel(params):
    return 418.9829 * len(params) - sum([param * math.sin(math.sqrt(abs(param))) for param in params])


def elipse(params):
    return sum([(10000 ** ((i - 1) / (len(params) - 1))) * (param ** 2) for i, param in enumerate(params)])


def cigar(params):
    return params[0] ** 2 + sum([10000 * param ** 2 for param in params[1:]])


def tablet(params):
    return 10000 * params[0] ** 2 + sum([param ** 2 for param in params])


@jit(nopython=True)
def sphere(params):
    total = 0

    for param in params:
        total += param ** 2

    return total
