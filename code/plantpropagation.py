import math
import random
import numpy as np

from environment import Environment
from point import Point


class PlantPropagation(object):
    """
    Python replication of the Plant Propagation algorithm as described in http://repository.essex.ac.uk/9974/1/paper.pdf
    """

    def __init__(self, bench, bounds, max_evaluations, m, max_runners, tanh_mod=1):
        """
        args:
            bench: the benchmark function object
            bounds: the boundaries of the bench
            max_evaluations (int): the maximum number of evaluations to run
            m (int): the population size
            max_runners (int): the maximum number of runners per individual
            tanh_mod (float): NOT IN ORIGINAL PAPER correction factor for tanh
            (makes distance at which good runners are generated smaller)
        """
        self.env = Environment(bounds, bench)

        self.population = self.env.get_random_population(m)

        self.iteration = 0
        self.max_evaluations = max_evaluations

        self.m = m

        self.max_runners = max_runners

        self.tanh_mod = tanh_mod

    def __repr__(self):
        return type(self).__name__ + \
                f'(bench={self.env.function.__name__}, \
                bounds={self.env.bounds}, \
                max_evaluations={self.max_evaluations}, \
                max_runners={self.max_runners}, \
                m={self.m}, \
                tanh_mod={self.tanh_mod})'

    def convert_fitness(self, fitness):
        """
        Converts fitness to a number between 0 and 1 (inclusive) where 0 is the
        current worst individual and 1 is the best.

        args:
            fitness (float): the fitness of an individual

        returns:
            A fitness value between 0 and 1 (inclusive).
        """
        return (self.z_max - fitness) / (self.z_max - self.z_min)

    @property
    def z_min(self):
        """
        returns:
            The fitness of the individual with the best fitness (lowest
            objective value). Assumes the population is sorted.
        """
        return self.population[0].fitness

    @property
    def z_max(self):
        """
        returns:
            The fitness of the individual with the worst fitness (highest
            objective value). Assumes the population is sorted.
        """
        return self.population[:self.m][-1].fitness

    def get_runner(self, pos, corr_fitness):
        """
        Get a runner from the given position with a distance calculated by the
        relative fitness.

        args:
            pos: the position of the parent
            corr_fitness (float): the corrected fitness of the parent

        returns:
            A point object.
        """
        distances = np.array([2 * (1 - corr_fitness) * (random.random() - 0.5) for _ in range(self.env.d)])

        scaled_dist = [(np.diff(self.env.bounds[i]) * distances[i])[0] for i in range(self.env.d)]
        runner = Point(self.env.limit_bounds(pos + scaled_dist), self.env)

        return runner

    def get_runners(self, plant):
        """
        Create all the children for this plant.

        returns:
            A list of Point objects (children).
        """
        runners = []

        if self.z_max - self.z_min > 0:
            corr_fitness = 0.5 * (math.tanh(4 * self.tanh_mod * self.convert_fitness(plant.fitness) - 2 * self.tanh_mod) + 1)
        else:
            corr_fitness = 0.5

        runners_amount = corr_fitness * self.max_runners * random.random()

        runners_amount = max(1, math.ceil(runners_amount))

        for _ in range(runners_amount):
            runner = self.get_runner(plant.pos, corr_fitness)
            runners.append(runner)

        return runners

    def start(self):
        """
        Starts the algorithm. Performs generations until the max number of
        evaluations is passed.

        Note that the algorithm always finishes a generation and can therefore
        complete more evaluations than defined.
        """
        while self.env.evaluation_number < self.max_evaluations:
            # Ascending sort + selection
            self.population = sorted(self.population, key=lambda plant: plant.fitness)[:self.m]

            # Create runners (children) for all plants
            for plant in self.population[:self.m]:
                self.population += self.get_runners(plant)

            self.iteration += 1
            self.env.generation_number += 1
