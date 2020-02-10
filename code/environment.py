import math
import numpy as np
from point import Point


class Environment(object):
    """
    The Environment class interfaces with the Point class in such a way that the
    algorithm class does not have to know anything about the benchmark function
    or its properties.
    """

    def __init__(self, bounds, bench):
        """
        args:
            bounds: the boundaries of the bench
            bench: the benchmark function object
        """
        self.d = len(bounds)

        self.bounds = bounds

        self.bench = bench

        # Prepare data lists for statistics
        self.evaluation_statistics = []
        self.evaluation_statistics_best = []
        self.generation_statistics = []

        self.generation_number = 0
        self.evaluation_number = 0
        self.cur_best = math.inf

    def get_random_population(self, N):
        """
        Randomly initializes a population of size N.

        args:
            N (int): the number of individuals to create

        returns:
            A list of Point objects.
        """
        return [self.get_random_point() for _ in range(N)]

    def get_random_point(self):
        """
        Create a (uniform) random individual.

        returns:
            A Point object.
        """
        pos = [np.random.uniform(self.bounds[i][0], self.bounds[i][1]) for i in range(self.d)]

        return Point(np.array(pos), self)

    def calculate_fitness(self, pos):
        """
        Calculate the fitness of an individual that is at a specific position.

        args:
            pos: a set of coordinates

        returns:
            The value of the bench on that position (float).
        """
        self.evaluation_number += 1
        fitness = self.bench(pos)
        self.evaluation_statistics.append(fitness)

        if fitness < self.cur_best:
            self.cur_best = fitness

        self.evaluation_statistics_best.append(self.cur_best)
        self.generation_statistics.append(self.generation_number)

        return fitness

    def limit_bounds(self, pos):
        """
        Truncate values of the set of coordinates to the bounds if the bounds
        of the bench are exceeded. This method is used in PPA and ultimately
        causes the bounds of the benchmark function to be examined more often.

        args:
            pos: a set of coordinates

        returns:
            The corrected position.
        """
        for i in range(self.d):
            lo_bound = self.bounds[i][0]
            hi_bound = self.bounds[i][1]

            pos[i] = lo_bound if pos[i] < lo_bound else hi_bound if pos[i] > hi_bound else pos[i]

        return pos

    def wrap_bounds(self, pos):
        """
        Wrap values of the set of coordinates if the bounds of the bench are
        exceeded. This method is used in FWA and tends to correct individuals
        to a position near the center of the two bounds.

        args:
            pos: a set of coordinates

        returns:
            The corrected position.
        """
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
