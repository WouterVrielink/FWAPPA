import math
import random
import numpy as np
import matplotlib.pyplot as plt

from environment import Environment
from point import Point

class PlantPropagation(object):
    """docstring for PlantPropagation."""

    def __init__(self, bench_function, bounds, max_evaluations, N, max_runners, m, tanh_mod=1):
        self.N = N

        self.env = Environment(bounds, bench_function)

        self.population = self.env.get_random_population(N)

        self.iteration = 0
        self.max_evaluations = max_evaluations

        self.m = m

        self.max_runners = max_runners

        self.tanh_mod = tanh_mod

    def __repr__(self):
        return type(self).__name__ + f'(N={self.N},' + \
                f'bounds={self.env.bounds},' + \
                f'bench_function={self.env.function.__name__},' + \
                f'max_iter={self.max_evaluations},' + \
                f'max_runners={self.max_runners},' + \
                f'm={self.m},' + \
                f'tanh_mod={self.tanh_mod})'

    def convert_fitness(self, fitness):
        return (self.z_max - fitness) / (self.z_max - self.z_min)

    @property
    def z_min(self):
        return self.population[0].fitness

    @property
    def z_max(self):
        return self.population[:self.m][-1].fitness

    def get_runner(self, pos, corr_fitness):
        distances = np.array([2 * (1 - corr_fitness) * (random.random() - 0.5) for _ in range(self.env.d)])

        scaled_dist = [(np.diff(self.env.bounds[i]) * distances[i])[0] for i in range(self.env.d)]
        runner = Point(self.env.limit_bounds(pos + scaled_dist), self.env)

        return runner

    def get_runners(self, plant):
        runners = []

        if self.z_max - self.z_min > 0:
            corr_fitness = 0.5 * (math.tanh(4 * self.tanh_mod * self.convert_fitness(plant.fitness) - 2 * self.tanh_mod) + 1)
            # corr_fitness = 0.5 * (math.tanh(16 * self.convert_fitness(plant.fitness) - 8) + 1)
        else:
            corr_fitness = 0.5

        runners_amount = corr_fitness * self.max_runners * random.random()

        runners_amount = max(1, math.ceil(runners_amount))

        for _ in range(runners_amount):
            runner = self.get_runner(plant.pos, corr_fitness)
            runners.append(runner)

        return runners

    def start(self):
        best = []
        fitness_avg = []

        while self.env.evaluation_number < self.max_evaluations:
            # plt.scatter([plant.pos[0] for plant in self.population], [plant.pos[1] for plant in self.population], c='c')

            # Ascending sort + selection
            self.population = sorted(self.population, key=lambda plant: plant.fitness)[:self.m]

            # fitness_avg.append(sum([plant.fitness for plant in self.population]) / len(self.population))

            # if not self.iteration % 1:
            #     plt.xlim(self.env.bounds[0])
            #     plt.ylim(self.env.bounds[1])
            #     plt.scatter([plant.pos[0] for plant in self.population], [plant.pos[1] for plant in self.population])
            #     plt.show()

            # Create runners (children) for all plants
            for plant in self.population[:self.m]:
                self.population += self.get_runners(plant)

            self.iteration += 1
            self.env.generation_number += 1

        #
        # plt.title('N: {} N_max: {} m: {} bench_function: {} dimensions: {}'.format(self.N, self.max_runners, self.m, self.env.function.__name__, self.env.d))
        # plt.plot(range(self.max_iterations), best, label='best')
        # plt.plot(range(self.max_iterations), fitness_avg, label='avg')
        # plt.xlabel('Iteration')
        # plt.ylabel('Benchmark score')
        # plt.legend()
        # plt.show()

        # print(self.population[0].pos)
        # print(self.population[0].fitness)
