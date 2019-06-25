"""
This file contains the Benchmark class, the set_benchmark_properties decorator,
the param_shift helper function, and the apply_add function.

The Benchmark class serves as a method to enable the setting and getting of
benchmark properties, while still acting as a function. In combination with the
set_benchmark_properties decorator this also enables the setting of various
parameters on function definition, having all the benefits of defining every
property only once in one place, and having none of the drawbacks of creating
huge class structures for each and every benchmark.

The param_shift and apply_add functions ensure that benchmarks are easily
translated.
"""


class Benchmark:
    """
    Benchmark function wrapper. Enables setting and getting of benchmark
    properties while also acting as a function.
    """

    def __init__(self, bounds, global_minima, func, **kwargs):
        """
        args:
            bounds: the boundaries of the bench
            global_minima: a list of the global minima in the function
            func: the actual benchmark function
            kwargs: other properties of the benchmark function
        """
        self.func = func

        self._bounds = bounds
        self._global_minima = global_minima

        self.kwargs = kwargs

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, *args, **kwargs):
        """
        Passes the arguments to the actual benchmark function. Makes sure the
        class acts as if it was the function.

        returns:
            The return value of the called function.
        """
        return self.func(*args, **kwargs)

    def __getattr__(self, attr):
        """
        args:
            attr: the attribute to get
        """
        return getattr(self.func, attr)

    @property
    def dims(self):
        """
        returns:
            The amount of dimensions of the bench (if set, otherwise raise).
        """
        if hasattr(self, '_dims'):
            return self._dims
        else:
            print('For N-dimensional benchmarks the property \'dims\' has to be set.')
            raise

    @dims.setter
    def dims(self, value):
        """
        args:
            value (int): the number of dimensions in the benchmark function
        """
        self._dims = value

    @property
    def bounds(self):
        """
        Changes the bounds to the proper dimensions if necessary.

        returns:
            The bounds of the bench.
        """
        return [self._bounds] * self.dims if self.is_n_dimensional else self._bounds

    @property
    def global_minima(self):
        """
        Changes the minima to the proper dimensions if necessary.

        returns:
            The global minima of the bench.
        """
        return [[minima for _ in range(self.dims)] for minima in self._global_minima] if self.is_n_dimensional else self._global_minima


def set_benchmark_properties(**kwargs):
    """
    Wrapper for the decorator. Makes sure all kwargs get passed to the
    benchmark object and the function is set as a parameter.

    returns:
        The decorator object that in turn returns a Benchmark object that acts
        as a function.
    """
    def decorator(func):
        return Benchmark(**kwargs, func=func)
    return decorator


def param_shift(params, value):
    """
    Shift the params by value.

    args:
        params: a list of coordinates
        value (float): the value to shift the params by

    returns:
        The params shifted by value.
    """
    if isinstance(value, tuple):
        return [param + value for param, value in zip(params, value)]
    return [param + value for param in params]


def apply_add(bench, value=10, name='_add'):
    """
    Function that creates a new bench object that has a new name and gets value
    added to each of the parameters that will be passed to the object.

    args:
        bench: the benchmark function object
        value (float): the value to shift the params by
        name (str): the string that gets added to the object name
    """
    # When value is a tuple or a list, each value will be applied seperately
    if isinstance(value, (tuple, list)):
        if len(value) != len(bench.bounds) or len(value) != len(bench.global_minima[0]):
            print("Value should be the same length as bounds and global_minima, or a scalar.")
            raise

        bounds = [(min - value, max - value) for (min, max), value in zip(bench.bounds, value)]
        global_minima = [[minval - value for minval, value in zip(minima, value)] for minima in bench.global_minima]
    # Otherwise value should be a scalar
    else:
        bounds = [(min - value, max - value) for (min, max) in bench.bounds]
        global_minima = [[minval - value for minval in minima] for minima in bench.global_minima]

    # Build the new function
    def new_func(params):
        params = param_shift(params, value)

        return bench(params)

    # Rename it
    new_func.__name__ = bench.__name__ + name + (str(value) if name == '_add' else '')

    # Make the object
    new_bench = Benchmark(func=new_func, bounds=bounds, global_minima=global_minima, **bench.kwargs)

    # For ease of use we want to carry this variable; we assume it is the same
    if hasattr(bench, '_dims'):
        new_bench.dims = bench.dims

    return new_bench
