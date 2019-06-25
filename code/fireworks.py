import math
import random
import numpy as np

from environment import Environment
from point import Point


class Fireworks(object):
    """
    Python replication of the algorithm described in https://www.researchgate.net/profile/Ying_Tan5/publication/220704568_Fireworks_Algorithm_for_Optimization/links/00b7d5281fc26a092a000000.pdf
    """

    def __init__(self, bench, bounds, max_evaluations, N, m, m_roof, a, b, max_amp):
        """
        args:
            bench: the benchmark function object
            bounds: the boundaries of the bench
            max_evaluations (int): the masimum number of evaluations to run
            N (int): the population size
            m (int): the maximum number of sparks
            m_roof (int): the maximum number of gaussian sparks
            a (int): the lower bound on the number of sparks
            b (int): the upper bound on the number of sparks
            max_amp (float): the maximum amplitude at which to generate sparks
        """
        self.N = N

        self.env = Environment(bounds, bench)

        self.population = self.env.get_random_population(N)

        self.iteration = 0
        self.max_evaluations = max_evaluations

        #  Max sparks and # Gaussian sparks
        self.m = m
        self.m_roof = m_roof

        # Lower and upper bound on sparks; a < b < 1
        self.am = a * self.m
        self.bm = b * self.m

        # Spark maximum amplitude
        self.max_amp = max_amp

    def __repr__(self):
        return type(self).__name__ + \
                f'bench={self.env.bench}, \
                bounds={self.env.bounds}, \
                max_evaluations={self.max_evaluations}, \
                N={self.N}, \
                m={self.m}, \
                m_roof={self.m_roof}, \
                a={self.am}, \
                b={self.bm}, \
                max_amp={self.max_amp})'

    @property
    def y_min(self):
        """
        returns:
            The fitness of the individual with the best fitness (lowest
            objective value). Assumes the population is sorted.
        """
        return self.population[0].fitness

    @property
    def y_max(self):
        """
        returns:
            The fitness of the individual with the worst fitness (highest
            objective value). Assumes the population is sorted.
        """
        return self.population[-1].fitness

    def get_sparks_amount(self, firework):
        """
        Method that calculates the number of children of an individual.

        args:
            firework: the individual

        returns:
            The number of children (int).
        """
        eps = np.finfo(float).eps
        total_diff = sum([self.y_max - firework.fitness for firework in self.population])
        sparks_amount = self.m * ((self.y_max - firework.fitness) + eps) / (total_diff + eps)

        if sparks_amount < self.am:
            sparks_amount = round(self.am)
        elif sparks_amount > self.bm:
            sparks_amount = round(self.bm)
        else:
            sparks_amount = round(sparks_amount)

        return sparks_amount

    def get_sparks_amplitude(self, firework):
        """
        Get the maximum distance at which children should be generated for this
        individual.

        args:
            firework: the individual

        returns:
            The maximum distance of a spark.
        """
        eps = np.finfo(float).eps
        total_diff = sum([firework.fitness - self.y_min for firework in self.population])
        sparks_amplitude = self.max_amp * ((firework.fitness - self.y_min) + eps) / (total_diff + eps)

        return sparks_amplitude

    def get_spark(self, pos, amplitude):
        """
        Get a spark from the given position with a distance of at most
        amplitude.

        args:
            pos: the position of the parent
            amplitude: the maximum mutation distance

        returns:
            A Point object.
        """
        # WARNING in the paper it said round, but their code effectively ceils!
        z = math.ceil(self.env.d * random.random())

        h = (random.random() - 0.5) * 2 * amplitude

        # Prevent changing of mutable object
        new_pos = [x for x in pos]

        # randomly select z dimensions of origin
        for dim in random.sample(range(self.env.d), z):
            # and mutate
            new_pos[dim] += h

        return Point(self.env.wrap_bounds(new_pos), self.env)

    def get_spark_gaussian(self, pos):
        """
        Get a spark from the given position through the Gaussian sparks method.

        args:
            pos: the position of the parent
            amplitude: the maximum mutation distance

        returns:
            A Point object.
        """
        # WARNING in the paper it said round, but their code effectively ceils!
        z = math.ceil(self.env.d * random.random())

        g = random.gauss(1.0, 1.0)

        # Prevent changing of mutable object
        new_pos = [x for x in pos]

        # randomly select z dimensions of origin
        for dim in random.sample(range(self.env.d), z):
            # and mutate
            new_pos[dim] *= g

        return Point(self.env.wrap_bounds(new_pos), self.env)

    def get_sparks(self):
        """
        Create all the children for the current generation.

        returns:
            A list of Point objects (the children of the current generation).
        """
        sparks = []

        # Normal sparks
        for firework in self.population:
            sparks_amount = self.get_sparks_amount(firework)
            sparks_amplitude = self.get_sparks_amplitude(firework)

            for _ in range(int(sparks_amount)):
                sparks.append(self.get_spark(firework.pos, sparks_amplitude))

        # Gaussian sparks random selection NOT SAMPLE
        for firework in np.random.choice(self.population, self.m_roof):
            sparks.append(self.get_spark_gaussian(firework.pos))

        return sparks

    def calculate_distance_population(self, population):
        """
        Calculate the distance to every individual in the population for every
        individual in the population.

        args:
            population: a list of individuals

        returns:
            A list of total distances for each individual in the population.
        """
        return [sum([firework.euclidean_distance(point) for point in population]) for firework in population]

    def start(self):
        """
        Starts the algorithm. Performs generations until the max number of
        evaluations is passed.

        Note that the algorithm always finishes a generation and can therefore
        complete more evaluations than defined.
        """
        while self.env.evaluation_number < self.max_evaluations:
            # Sort the population Ascending
            self.population = sorted(self.population, key=lambda firework: firework.fitness)

            # Selection
            if len(self.population) > self.N:
                new_pop = [self.population[0]]

                dist = self.calculate_distance_population(self.population[1:])
                probabilities = dist / sum(dist)

                new_pop += list(np.random.choice(self.population[1:], self.N - 1, p=probabilities))

                self.population = new_pop

            # Create sparks (children)
            self.population += self.get_sparks()

            self.iteration += 1
            self.env.generation_number += 1
