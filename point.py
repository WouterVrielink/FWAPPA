class Point(object):
    """docstring for Point."""
    def __init__(self, pos, env):
        self.pos = pos
        self.env = env

        self.fitness_calculated = False
        self._fitness = None

    @property
    def fitness(self):
        if not self.fitness_calculated:
            self._fitness = self.env.calculate_fitness(self.pos)
            self.fitness_calculated = True
        return self._fitness

    def euclidean_distance(self, point):
        return sum([abs(self.pos[i] - point.pos[i]) for i in range(self.env.d)])

    def euclidean_distance_population(self, population):
        return sum([self.euclidean_distance(point) for point in population])
