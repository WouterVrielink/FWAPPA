from ppa import PlantPropagation
from fireworks import Fireworks
import helper_tools
import benchmarks

import json
import os

def load_config(file):
    with open(file, 'r') as f:
        config = json.load(f)
    return config


def do_run(alg, bench_function, bounds, max_evaluations, reps, verbose=1, version="DEFAULT"):
    config = load_config(f'config_{alg.__name__}.json')[version]

    if verbose:
        print("--------------------------------------")
        print(f"Running {alg.__name__} on {bench_function.__name__}...")

    for repetition in range(1, reps + 1):
        filename = helper_tools.get_name(alg, bench_function, version, len(bounds), repetition)

        if os.path.isfile(filename):
            print(f"\tRepetition {repetition} / {reps} - exists") if verbose else _
            continue

        print(f"\tRepetition {repetition} / {reps} - running") if verbose else _
        alg_instance = alg(bench_function, bounds, max_evaluations, *list(config.values()))
        alg_instance.start()

        print(f"\tRepetition {repetition} / {reps} - saving") if verbose else _
        helper_tools.save_to_csv(alg_instance, filename)


if __name__ == "__main__":
    evaluations = 10000
    dims = 2

    do_run(PlantPropagation, benchmarks.sphere, [(-100, 100) for _ in range(dims)], evaluations, 10)

    do_run(Fireworks, benchmarks.sphere, [(-100, 100) for _ in range(dims)], evaluations, 10)

    # TODO
    # helper_tools.save_to_csv()

    # TODO voor fireworks alles fixen
    # dims eruit slopen
