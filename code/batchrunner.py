"""
This file holds all the code required to run all experiments.

Running this file will result in a /data folder with folders organized as
follows:
data/<algorithm>_<configuration>/<benchmark variant>/<dimension>/

Two types of file will be made:
<run number>.csv
times.csv

The first file will contain the value obtained, the current best, and the
generation number of the algorithm that was applied to the benchmark for search
evaluation that is made.

The second file will contain the time each of the runs took. Keep in mind that
although computing time is registered, it does not account for any background
tasks that a system runs and is therefore very noisy.

Do not worry about interrupting the program, it will restart where it was based
on the output files that are in the /data folder.
"""

import json
import os

import numpy as np
from timeit import default_timer as timer

import code.helper_tools


def load_config(file):
    """
    Loads a config file (JSON) into a dictionary.

    args:
        file (str): the file that should be loaded

    returns:
        A config dictionary.
    """
    with open(file, 'r') as f:
        config = json.load(f)
    return config


def do_run(alg, bench, max_evaluations, reps, bounds=None, dims=2, prefix=None, version="DEFAULT", verbose=True):
    """
    Performs a run with the given parameters. Automatically saves the results in
    the proper folders.

    args:
        alg: the algorithm object
        bench: the benchmark function object
        max_evaluations (int): the maximum amount of evaluations to run
        reps (int): the amount of times an experiment should be repeated
        bounds: a tuple containing the bounds to set for the bench
        dims (int): the amount of dimensions in the bench
        prefix (str): an optional prefix for the experiment name
        version (str): the name of the parameter-set
        verbose (bool): whether the function should be verbose
    """
    if not bounds:
        bounds = bench.bounds

    config = load_config(f'configs/config_{alg.__name__}.json')[version]

    if verbose:
        print("--------------------------------------")
        print(f"Running {alg.__name__} on {bench.__name__} in {dims}D...")

    filename_time = helper_tools.get_time_name(alg, bench, version, dims, prefix)

    for repetition in range(1, reps + 1):
        filename_stats = helper_tools.get_name(alg, bench, version, dims, repetition, prefix)

        if os.path.isfile(filename_stats):
            print(f"\tRepetition {repetition} / {reps} - exists") if verbose else _
            continue

        alg_instance = alg(bench, bounds, max_evaluations, *list(config.values()))

        print(f"\tRepetition {repetition} / {reps} - running") if verbose else _
        start = timer()
        alg_instance.start()
        end = timer()

        print(f"\tRepetition {repetition} / {reps} - saving") if verbose else _
        helper_tools.save_to_csv(alg_instance, filename_stats)
        helper_tools.save_time(end - start, alg_instance.env.evaluation_number, repetition, filename_time)


if __name__ == "__main__":
    import benchmark
    import benchmark_functions as benchmarks

    from plantpropagation import PlantPropagation
    from fireworks import Fireworks

    evaluations = 10000
    repetitions = 10
    maxDims = 100

    # Prepare globals
    bench_fun = [getattr(benchmarks, fun) for fun in dir(benchmarks) if hasattr(getattr(benchmarks, fun), 'is_n_dimensional')]
    two_dim_fun = [fun for fun in bench_fun if not fun.is_n_dimensional]
    n_dim_fun = [fun for fun in bench_fun if fun.is_n_dimensional]

    non_center_two_dim_fun = [fun for fun in two_dim_fun if (0, 0) not in fun.global_minima]
    non_center_n_dim_fun = [fun for fun in n_dim_fun if (0) not in fun._global_minima]

    algorithms = (Fireworks, PlantPropagation)

    # 2-dimensional
    for alg in algorithms:
        for bench in two_dim_fun:
            do_run(alg, bench, evaluations, repetitions)

        for bench in non_center_two_dim_fun:
            bench_center = benchmark.apply_add(bench, value=bench.global_minima[0], name='_center')

            do_run(alg, bench_center, evaluations, repetitions)

            for value in (0.1, 1, 10, 100, 1000):
                bench_add = benchmark.apply_add(bench_center, value=value)

                do_run(alg, bench_add, evaluations, repetitions)

    # N-dimensional
    for alg in algorithms:
        for dims in range(2, maxDims + 1):
            for bench in n_dim_fun:
                bench.dims = dims

                do_run(alg, bench, evaluations, repetitions, dims=dims)

                bench_add = benchmark.apply_add(bench)

                do_run(alg, bench_add, evaluations, repetitions, dims=dims)

            for bench in non_center_n_dim_fun:
                bench.dims = dims

                bench_center = benchmark.apply_add(bench, value=bench.global_minima[0], name='_center')

                do_run(alg, bench_center, evaluations, repetitions, dims=dims)

    # 2D 21*21 uniform grid over full domain for all benchmarks
    grid_number = 21

    for alg in algorithms:
        for bench in bench_fun:
            bench.dims = 2

            xpositions = np.linspace(*bench.bounds[0], num=grid_number)
            ypositions = np.linspace(*bench.bounds[1], num=grid_number)

            for x_i, x in enumerate(xpositions):
                for y_i, y in enumerate(ypositions):
                    bench_add = benchmark.apply_add(bench, value=(x, y), name=f'_{x_i}_{y_i}')

                    # Once for shifted domain and once for non-shifted domain
                    do_run(alg, bench_add, evaluations, repetitions, prefix='shifted_domain')
                    do_run(alg, bench_add, evaluations, repetitions, prefix='unshifted_domain', bounds=bench.bounds)
