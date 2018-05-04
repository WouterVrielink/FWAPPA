import numpy as np
from point import Point

class Environment(object):
    """docstring for Environment."""
    def __init__(self, d, bounds, function):
        self.d = d

        self.bounds = bounds

        self.function = function

    def get_random_population(self, N):
        return [self.get_random_point() for _ in range(N)]

    def get_random_point(self):

        pos = [np.random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.d)]

        return Point(np.array(pos), self)

    def calculate_fitness(self, pos):
        return self.function(pos)

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

            # DAAN
            if not (lo_bound <= pos[i] <= hi_bound):
                pos[i] = lo_bound + abs(pos[i]) % (hi_bound - lo_bound)

        return pos
