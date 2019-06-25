import math
import random
import numpy as np

from environment import Environment
from point import Point


class Fireworks(object):
    """docstring for Fireworks."""

    def __init__(self, bench_function, bounds, max_evaluations, N, m, m_roof, a, b, max_amp):
        self.N = N

        self.env = Environment(bounds, bench_function)

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
                f'bench_function={self.env.bench_function}, \
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
        return self.population[0].fitness

    @property
    def y_max(self):
        return self.population[-1].fitness

    def get_sparks_amount(self, firework):
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
        eps = np.finfo(float).eps
        total_diff = sum([firework.fitness - self.y_min for firework in self.population])
        sparks_amplitude = self.max_amp * ((firework.fitness - self.y_min) + eps) / (total_diff + eps)

        return sparks_amplitude

    def get_spark(self, pos, amplitude):
        # WARNING in paper staat round, maar in code doen ze effectief ceil
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
        # WARNING in paper staat round, maar in code doen ze effectief ceil
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
        return [sum([firework.euclidean_distance(point) for point in population]) for firework in population]

    def start(self):
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
