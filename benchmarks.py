import math
import numpy as np


class Benchmark:
    def __init__(self, bounds, global_minima, func, **kwargs):
        self.func = func

        self._bounds = bounds
        self._global_minima = global_minima

        self.kwargs = kwargs

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __getattr__(self, attr):
        return getattr(self.func, attr)

    @property
    def dims(self):
        if hasattr(self, '_dims'):
            return self._dims
        else:
            print('For N-dimensional benchmarks the property \'dims\' has to be set.')
            raise

    @dims.setter
    def dims(self, value):
        self._dims = value

    @property
    def bounds(self):
        return [self._bounds] * self.dims if self.is_n_dimensional else self._bounds

    @property
    def global_minima(self):
        return [[minima for _ in range(self.dims)] for minima in self._global_minima] if self.is_n_dimensional else self._global_minima


def set_benchmark_properties(**kwargs):
    def decorator(func):
        return Benchmark(**kwargs, func=func)
    return decorator


def param_shift(params, value):
    if isinstance(value, tuple):
        return [param + value for param, value in zip(params, value)]
    return [param + value for param in params]


def apply_add(bench_function, value=10, name='_add'):
    # When value is a tuple or a list, each value will be applied seperately
    if isinstance(value, (tuple, list)):
        if len(value) != len(bench_function.bounds) or len(value) != len(bench_function.global_minima[0]):
            print("Value should be the same length as bounds and global_minima, or a scalar.")
            raise
        bounds = [(min - value, max - value) for (min, max), value in zip(bench_function.bounds, value)]

        global_minima = [[minval - value for minval, value in zip(minima, value)] for minima in bench_function.global_minima]

    # Otherwise value should be a scalar
    else:
        bounds = [(min - value, max - value) for (min, max) in bench_function.bounds]
        global_minima = [[minval - value for minval in minima] for minima in bench_function.global_minima]

    # Build the new function
    def new_func(params):
        params = param_shift(params, value)

        return bench_function(params)

    # Rename it
    new_func.__name__ = bench_function.__name__ + name + (str(value) if name == '_add' else '')

    new_bench = Benchmark(func=new_func, bounds=bounds, global_minima=global_minima, **bench_function.kwargs)

    # For ease of use we want to carry this variable; we assume it is the same
    if hasattr(bench_function, '_dims'):
        new_bench.dims = bench_function.dims

    return new_bench


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
    return sum([(10000 ** ((i - 1) / (len(params) - 1))) * (param ** 2) for i, param in enumerate(params)])


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
    return 10000 * params[0] ** 2 + sum([param ** 2 for param in params])


@set_benchmark_properties(
    official_name='Sphere',
    is_n_dimensional=True,
    bounds=(-100, 100),
    correction=0,
    global_minima=[(0)])
def sphere(params):
    return sum([param ** 2 for param in params])
