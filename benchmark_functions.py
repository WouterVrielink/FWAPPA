"""
This file contains 14 different benchmark functions. Each of the benchmark
functions has a wrapper that enables the setting of benchmark properties.

In total, there are five 2-D functions, and nine N-D functions (N => 2).
"""

import math
import numpy as np
from benchmark import set_benchmark_properties


@set_benchmark_properties(
    official_name='Six-Hump-camel',
    is_n_dimensional=False,
    bounds=[(-3, 3), (-2, 2)],
    correction=-1.031628453489877,
    global_minima=[(0.0898, -0.7126), (-0.0898, 0.7126)])
def six_hump_camel(params):
    first_term = (4 - 2.1 * (params[0] ** 2) + (params[0] ** 4) / 3) * params[0] ** 2
    second_term = params[0] * params[1]
    third_term = (-4 + 4 * (params[1] ** 2)) * params[1] ** 2

    return first_term + second_term + third_term


@set_benchmark_properties(
    official_name='Martin-Gaddy',
    is_n_dimensional=False,
    bounds=[(-20, 20), (-20, 20)],
    correction=0,
    global_minima=[(5, 5)])
def martin_gaddy(params):
    first_term = (params[0] - params[1]) ** 2
    second_term = ((params[0] + params[1] - 10) / 3) ** 2

    return first_term + second_term


@set_benchmark_properties(
    official_name='Goldstein-Price',
    is_n_dimensional=False,
    bounds=[(-2, 2), (-2, 2)],
    correction=3,
    global_minima=[(0, -1)])
def goldstein_price(params):
    first_term = 1 + ((params[0] + params[1] + 1) ** 2) * (19 - 14 * params[0] + 3 * params[0] ** 2 - 14 * params[1] + 6 * params[0] * params[1] + 3 * params[1] ** 2)
    second_term = 30 + ((2 * params[0] - 3 * params[1]) ** 2) * (18 - 32 * params[0] + 12 * params[0] ** 2 + 48 * params[1] - 36 * params[0] * params[1] + 27 * params[1] ** 2)

    return first_term * second_term


@set_benchmark_properties(
    official_name='Branin',
    is_n_dimensional=False,
    bounds=[(-5, 15), (-5, 15)],
    correction=0.39788735772973816,
    global_minima=[(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)])
def branin(params):
    first_term = params[1] - (5.1 / (4 * math.pi ** 2)) * params[0] ** 2 + (5 / math.pi) * params[0] - 6
    second_term = 10 * (1 - 1 / (8 * math.pi)) * math.cos(params[0])

    return first_term ** 2 + second_term + 10


@set_benchmark_properties(
    official_name='Easom',
    is_n_dimensional=False,
    bounds=[(-100, 100), (-100, 100)],
    correction=-1,
    global_minima=[(math.pi, math.pi)])
def easom(params):
    return -math.cos(params[0]) * math.cos(params[1]) * math.exp(-(params[0] - math.pi) ** 2 - (params[1] - math.pi) ** 2)


@set_benchmark_properties(
    official_name='Rosenbrock',
    is_n_dimensional=True,
    bounds=(-5, 10),
    correction=0,
    global_minima=[(1)])
def rosenbrock(params):
    return sum([100 * (params[i + 1] - params[i] ** 2) ** 2 + (params[i] - 1) ** 2 for i in range(len(params) - 1)])


@set_benchmark_properties(
    official_name='Ackley',
    is_n_dimensional=True,
    bounds=(-100, 100),
    correction=0,
    global_minima=[(0)])
def ackley(params):
    first_term = -20 * math.exp(-0.2 * math.sqrt((1 / len(params)) * sum([param ** 2 for param in params])))
    second_term = math.exp((1 / len(params)) * sum([math.cos(2 * math.pi * param) for param in params]))

    return first_term - second_term + 20 + math.e


@set_benchmark_properties(
    official_name='Griewank',
    is_n_dimensional=True,
    bounds=(-600, 600),
    correction=0,
    global_minima=[(0)])
def griewank(params):
    first_term = sum([(param ** 2) / 4000 for param in params])
    second_term = np.prod([math.cos(param / math.sqrt(i + 1)) for i, param in enumerate(params)])

    return 1 + first_term + second_term


@set_benchmark_properties(
    official_name='Rastrigrin',
    is_n_dimensional=True,
    bounds=(-5.12, 5.12),
    correction=0,
    global_minima=[(0)])
def rastrigrin(params):
    return 10 * len(params) + sum([param ** 2 - 10 * math.cos(2 * math.pi * param) for param in params])


@set_benchmark_properties(
    official_name='Schwefel',
    is_n_dimensional=True,
    bounds=(-500, 500),
    correction=0,
    global_minima=[(420.9687)])
def schwefel(params):
    return 418.9829 * len(params) - sum([param * math.sin(math.sqrt(abs(param))) for param in params])


@set_benchmark_properties(
    official_name='Elipse',
    is_n_dimensional=True,
    bounds=(-100, 100),
    correction=0,
    global_minima=[(0)])
def elipse(params):
    return sum([(10000 ** (i / (len(params) - 1))) * (param ** 2) for i, param in enumerate(params)])


@set_benchmark_properties(
    official_name='Cigar',
    is_n_dimensional=True,
    bounds=(-100, 100),
    correction=0,
    global_minima=[(0)])
def cigar(params):
    return params[0] ** 2 + sum([10000 * param ** 2 for param in params[1:]])


@set_benchmark_properties(
    official_name='Tablet',
    is_n_dimensional=True,
    bounds=(-100, 100),
    correction=0,
    global_minima=[(0)])
def tablet(params):
    return 10000 * params[0] ** 2 + sum([param ** 2 for param in params[1:]])


@set_benchmark_properties(
    official_name='Sphere',
    is_n_dimensional=True,
    bounds=(-100, 100),
    correction=0,
    global_minima=[(0)])
def sphere(params):
    return sum([param ** 2 for param in params])
