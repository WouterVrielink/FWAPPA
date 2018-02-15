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
        return self._fitness
