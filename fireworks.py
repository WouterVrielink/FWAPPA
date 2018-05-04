import random

class Fireworks(object):
    """docstring for Fireworks."""

    def __init__(self, N, d, bounds, bench_function, max_iter):
        self.N = N
        self.bench_function = bench_function

        self.env = Environment(d, bounds, bench_function)

        self.population = self.env.get_random_population(N)

        self.iteration = 0
        self.max_iterations = max_iter

        # Spark control
        self.m = m
        self.m_roof = m_roof

        # Lower and upper bound on sparks; a < b < 1
        self.am = a * self.m
        self.bm = b * self.m

        # Spark maximum amplitude
        self.max_amp = max_amp


    @property
    def y_min(self):
        return self.population[0].fitness

    @property
    def y_max(self):
        return self.population[-1].fitness

    def get_sparks_amount(self, firework):
        total_diff = sum([self.y_max - firework.fitness for firework in self.population])

        # DAAN gekke notatie
        if total_diff > 0:
            sparks_amount = self.m * (self.y_max - firework.fitness) / total_diff
        else:
            sparks_amount = self.m / self.N
            # sparks_amount = 0

        if sparks_amount < self.am:
            sparks_amount = round(self.am)
        elif sparks_amount > self.bm:
            sparks_amount = round(self.bm)
        else:
            sparks_amount = round(sparks_amount)

        return sparks_amount

    def get_sparks_amplitude(self, firework):
        total_diff = sum([firework.fitness - self.y_min for firework in self.population])

        # DAAN Amp == 0??
        if total_diff > 0:
            sparks_amplitude = max_amp * (firework.fitness - self.y_min) / total_diff
        else:
            sparks_amplitude = 0

        return sparks_amplitude



    def get_spark(self, pos, amplitude):
        z = round(self.env.d * random.random)

        h = (random.random() - 0.5) * 2 * amplitude

        # Prevent changing of mutable object
        new_pos = [x for x in pos]

        # randomly select z dimensions of origin
        for random in random.sample(range(self.env.d), z):
            # and mutate
            new_pos[random] += h

        return Point(self.env.wrap_bounds(new_pos), self.env)

    def get_spark_gaussian(self, pos):
        z = round(self.env.d * random.random)

        g = random.gauss(1.0, 1.0)

        # Prevent changing of mutable object
        new_pos = [x for x in pos]

        # randomly select z dimensions of origin
        for random in random.sample(range(self.env.d), z):
            # and mutate
            new_pos[random] *= g

        return Point(self.correct_bounds(new_pos), self.env)

    def get_sparks(self):
        sparks = []

        # Normal sparks
        for firework in self.population:
            sparks_amount = get_sparks_amount(firework)
            sparks_amplitude = get_sparks_amplitude(firework)

            for _ in range(sparks_amount):
                sparks.append(get_spark(firework.pos, sparks_amplitude))

        # Gaussian sparks DAAN random selection of sample?
        for firework in random.sample(self.population, self.m_roof):
            sparks.append(get_spark_gaussian(firework.poss))

        return sparks

    def start(self):
        best = []
        fitness_avg = []

        while self.iteration < self.max_iterations:
            # Selection
            if len(self.population) > self.n:
                # DAAN selection TODO
                # new_pop = [self.population[0]]

                # TODO NOG FOUT
                self.population = sorted(self.population, key=lambda firework: firework.fitness)
                self.population = self.population[:self.N]

            # Sort the population Ascending
            self.population = sorted(self.population, key=lambda firework: firework.fitness)

            # Create sparks (children)
            self.population += get_sparks()

            self.iteration += 1
