from ppa import PlantPropagation
from fireworks import Fireworks
import helper_tools
import benchmarks

from timeit import default_timer as timer
import json
import os

def load_config(file):
    with open(file, 'r') as f:
        config = json.load(f)
    return config


def do_run(alg, bench_function, bounds, max_evaluations, reps, verbose=1, version="DEFAULT"):
    config = load_config(f'configs/config_{alg.__name__}.json')[version]

    if verbose:
        print("--------------------------------------")
        print(f"Running {alg.__name__} on {bench_function.__name__} in {len(bounds)}D...")

    for repetition in range(1, reps + 1):
        filename_stats = helper_tools.get_name(alg, bench_function, version, len(bounds), repetition)
        filename_time = helper_tools.get_time_name(alg, bench_function, version, len(bounds))

        if os.path.isfile(filename_stats):
            print(f"\tRepetition {repetition} / {reps} - exists") if verbose else _
            continue

        print(f"\tRepetition {repetition} / {reps} - running") if verbose else _
        alg_instance = alg(bench_function, bounds, max_evaluations, *list(config.values()))
        start = timer()
        alg_instance.start()
        end = timer()

        print(f"\tRepetition {repetition} / {reps} - saving") if verbose else _
        helper_tools.save_to_csv(alg_instance, filename_stats)
        helper_tools.save_time(end - start, alg_instance.env.evaluation_number, repetition, filename_time)

if __name__ == "__main__":
    evaluations = 10000

    # 2-dimensional
    dims = 2
    for alg in (PlantPropagation, Fireworks):
        do_run(alg, benchmarks.easom, [(-100, 100) for _ in range(dims)], evaluations, 10)

        do_run(alg, benchmarks.branin, [(-5, 15) for _ in range(dims)], evaluations, 10)

        do_run(alg, benchmarks.goldstein_price, [(-2, 2) for _ in range(dims)], evaluations, 10)

        do_run(alg, benchmarks.martin_gaddy, [(-20, 20) for _ in range(dims)], evaluations, 10)

        do_run(alg, benchmarks.six_hump_camel, [(-3, 3), (-2, 2)], evaluations, 10)

    # N-dimensional
    for dims in range(2, 101):
        for alg in (PlantPropagation, Fireworks):
            do_run(alg, benchmarks.sphere, [(-100, 100) for _ in range(dims)], evaluations, 10)

            do_run(alg, benchmarks.tablet, [(-100, 100) for _ in range(dims)], evaluations, 10)

            do_run(alg, benchmarks.cigar, [(-100, 100) for _ in range(dims)], evaluations, 10)

            do_run(alg, benchmarks.elipse, [(-100, 100) for _ in range(dims)], evaluations, 10)

            do_run(alg, benchmarks.schwefel, [(-500, 500) for _ in range(dims)], evaluations, 10)

            do_run(alg, benchmarks.rastigrin, [(-5.12, 5.12) for _ in range(dims)], evaluations, 10)

            do_run(alg, benchmarks.ackley, [(-100, 100) for _ in range(dims)], evaluations, 10)

            do_run(alg, benchmarks.rosenbrock, [(-5, 10) for _ in range(dims)], evaluations, 10)
