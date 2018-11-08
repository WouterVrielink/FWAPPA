from ppa import PlantPropagation
from fireworks import Fireworks
import helper_tools
import benchmarks

from timeit import default_timer as timer
import json
import os
import math


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
    repetitions = 10

    # 2-dimensional
    for alg in (PlantPropagation, Fireworks):
        for bench_function, domain in benchmarks.two_dim_bench_functions().items():
            do_run(alg, bench_function, domain, evaluations, repetitions)

            for value in (0.1, 1, 10, 100, 1000):
                bench_function_add, domain_add = benchmarks.apply_add(bench_function, domain, value=value)

                do_run(alg, bench_function_add, domain_add, evaluations, repetitions)

        # Centered easom...
        bench_function, domain = benchmarks.apply_add(benchmarks.easom, [(-100, 100), (-100, 100)], value=-math.pi, name='_center')
        do_run(alg, bench_function, domain, evaluations, repetitions)

    # N-dimensional
    for dims in range(2, 101):
        for alg in (PlantPropagation, Fireworks):
            for bench_function, domain in benchmarks.n_dim_bench_functions().items():
                domain = [domain for _ in range(dims)]
                do_run(alg, bench_function, domain, evaluations, repetitions)

                bench_function_add, domain_add = benchmarks.apply_add(bench_function, domain)

                do_run(alg, bench_function_add, domain_add, evaluations, repetitions)
