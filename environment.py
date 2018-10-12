import math
import numpy as np
from point import Point

class Environment(object):
    """docstring for Environment."""
    def __init__(self, bounds, function):
        self.d = len(bounds)

        self.bounds = bounds

        self.function = function

        self.evaluation_statistics = []
        self.evaluation_statistics_best = []
        self.generation_statistics = []

        self.generation_number = 0
        self.evaluation_number = 0
        self.cur_best = math.inf

    def get_random_population(self, N):
        return [self.get_random_point() for _ in range(N)]

    def get_random_point(self):
        pos = [np.random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.d)]

        return Point(np.array(pos), self)

    def calculate_fitness(self, pos):
        self.evaluation_number += 1
        fitness = self.function(pos)
        self.evaluation_statistics.append(fitness)

        if fitness < self.cur_best:
            self.cur_best = fitness

        self.evaluation_statistics_best.append(self.cur_best)
        self.generation_statistics.append(self.generation_number)

        return fitness

    def limit_bounds(self, distances):
        for i in range(self.d):
            lo_bound = self.bounds[i][0]
            hi_bound = self.bounds[i][1]

            distances[i] = lo_bound if distances[i] < lo_bound else hi_bound if distances[i] > hi_bound else distances[i]

        return distances

    def wrap_bounds(self, pos):
        for i in range(self.d):
            lo_bound = self.bounds[i][0]
            hi_bound = self.bounds[i][1]

            if not (lo_bound <= pos[i] <= hi_bound):
                pos[i] = lo_bound + abs(pos[i]) % (hi_bound - lo_bound)

        return pos

    def get_evaluation_statistics(self):
        return list(range(1, self.evaluation_number + 1)), self.evaluation_statistics

    def get_evaluation_statistics_best(self):
        return list(range(1, self.evaluation_number + 1)), self.evaluation_statistics_best

    def get_generation_statistics(self):
        return list(range(1, self.evaluation_number + 1)), self.generation_statistics
