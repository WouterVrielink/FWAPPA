import math
import numpy as np


def six_hump_camel(params):
    assert len(params) == 2, \
        "Dimensions are incorrect."

    assert -3 <= params[0] <= 3, \
        "x_1 is out of bounds with value: {}".format(params[0])

    assert -2 <= params[1] <= 2, \
        "x_2 is out of bounds with value: {}".format(params[1])

    first_term = (4 - 2.1 * (params[0] ** 2) + (params[0] ** 4) / 3) * params[0] ** 2
    second_term = params[0] * params[1]
    third_term = (-4 + 4 * (params[1] ** 2)) * params[1] ** 2

    return first_term + second_term + third_term


def martin_gaddy(params):
    assert len(params) == 2, \
        "Dimensions are incorrect."

    assert -20 <= params[0] <= 20, \
        "x_1 is out of bounds with value: {}".format(params[0])

    assert -20 <= params[1] <= 20, \
        "x_2 is out of bounds with value: {}".format(params[1])

    first_term = (params[0] - params[1]) ** 2
    second_term = ((params[0] + params[1] - 10) / 3) ** 2

    return first_term + second_term


def goldstein_price(params):
    assert len(params) == 2, \
        "Dimensions are incorrect."

    assert -2 <= params[0] <= 2, \
        "x_1 is out of bounds with value: {}".format(params[0])

    assert -2 <= params[1] <= 2, \
        "x_2 is out of bounds with value: {}".format(params[1])

    first_term = 1 + ((params[0] + params[1] + 1) ** 2) * (19 - 14 * params[0] + 3 * params[0] ** 2 - 14 * params[1] + 6 * params[0] * params[1] + 3 * params[1] ** 2)
    second_term = 30 + ((2 * params[0] - 3 * params[1]) ** 2) * (18 - 32 * params[0] + 12 * params[0] ** 2 + 48 * params[1] - 36 * params[0] * params[1] + 27 * params[1] ** 2)

    return first_term * second_term


def branin(params):
    assert len(params) == 2, \
        "Dimensions are incorrect."

    assert -5 <= params[0] <= 15, \
        "x_1 is out of bounds with value: {}".format(params[0])

    assert -5 <= params[1] <= 15, \
        "x_2 is out of bounds with value: {}".format(params[1])

    first_term = params[1] - (5.1 / (4 * math.pi ** 2)) * params[0] ** 2 + (5 / math.pi) * params[0] - 6
    second_term = 10 * (1 - 1 / (8 * math.pi)) * math.cos(params[0])

    return first_term ** 2 + second_term + 10


def easom(params):
    assert len(params) == 2, \
        "Dimensions are incorrect."

    assert -100 <= params[0] <= 100, \
        "x_1 is out of bounds with value: {}".format(params[0])

    assert -100 <= params[1] <= 100, \
        "x_2 is out of bounds with value: {}".format(params[1])

    return -math.cos(params[0]) * math.cos(params[1]) * math.exp(-(params[0] - math.pi) ** 2 - (params[1] - math.pi) ** 2)


def check_dims(params, lower, upper):
    for i, param in enumerate(params):
        assert lower <= param <= upper, \
            "x_{} is out of bounds with value: {}".format(i + 1, params[i])

    return True


def rosenbrock(params):
    assert check_dims(params, -5, 10)

    return sum([100 * (params[i + 1] - params[i] ** 2) ** 2 + (params[i] - 1) ** 2 for i in range(len(params) - 1)])


def ackley(params):
    assert check_dims(params, -100, 100)

    first_term = -20 * math.exp(-0.2 * math.sqrt((1 / len(params)) * sum([param ** 2 for param in params])))
    second_term = math.exp((1 / len(params)) * sum([math.cos(2 * math.pi * param) for param in params]))

    return first_term - second_term + 20 + math.e


def griewank(params):
    assert check_dims(params, -600, 600)

    first_term = sum([(param ** 2) / 4000 for param in params])
    second_term = np.prod([param / math.sqrt(i) for i, param in enumerate(params)])

    return 1 + first_term + second_term


def rastrigrin(params):
    assert check_dims(params, -5.12, 5.12)

    return 10 * len(params) + sum([param ** 2 - 10 * math.cos(2 * math.pi * param) for param in params])


def schwefel(params):
    assert check_dims(params, -500, 500)

    return 418.9823 * len(params) - sum([param * math.sin(math.sqrt(abs(param))) for param in params])


def elipse(params):
    assert check_dims(params, -100, 100)

    return sum([(10000 ** ((i - 1) / (len(params) - 1))) * (param ** 2) for i, param in enumerate(params)])


def cigar(params):
    assert check_dims(params, -100, 100)

    return params[0] ** 2 + sum([10000 * param ** 2 for param in params[1:]])


def tablet(params):
    assert check_dims(params, -100, 100)

    return 10000 * params[0] ** 2 + sum([param ** 2 for param in params])


def sphere(params):
    assert check_dims(params, -100, 100)

    return np.sum(np.array([param ** 2 for param in params]))
