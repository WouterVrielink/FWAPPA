class Point(object):
    """
    Provides a coordinate class that can calculate distances and determine if
    it was already once evaluated.
    """
    def __init__(self, pos, env):
        """
        args:
            pos: a list of coordinates
            env: the environment the point resides in
        """
        self.pos = pos
        self.env = env

        self._fitness = None

    @property
    def fitness(self):
        """
        Calculates the fitness of the individual if it was not already
        calculated.

        returns:
            The fitness of an individual
        """
        if self._fitness is None:
            self._fitness = self.env.calculate_fitness(self.pos)
        return self._fitness

    def euclidean_distance(self, point):
        """
        Calculates the difference between this point and another.

        args:
            point: a point object

        returns:
            The euclidean distance between the two points.
        """
        return sum([abs(self.pos[i] - point.pos[i]) for i in range(self.env.d)])
