import csv
import os
import itertools as it
import numpy as np
import matplotlib.pyplot as plt

def build_path(alg, bench_function, version, dims):
    return f'data/{alg.__name__}_{version}/{bench_function.__name__}/{dims}d'

def get_name(alg, bench_function, version, dims, repetition):
    return f'{build_path(alg, bench_function, version, dims)}/{str(repetition)}.csv'

def get_time_name(alg, bench_function, version, dims):
    return f'{build_path(alg, bench_function, version, dims)}/time.csv'

def save_to_csv(alg, filename):
    x, y = alg.env.get_evaluation_statistics()
    _, best_y = alg.env.get_evaluation_statistics_best()
    _, generations = alg.env.get_generation_statistics()

    # Check if folder exists, else make it
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(filename, mode='w') as file:
        writer = csv.writer(file)

        writer.writerow(['evaluation', 'value', 'curbest', 'generation'])

        for row in zip(x, y, best_y, generations):
            writer.writerow(row)

def save_time(time, total_evals, rep, filename):
    # Check if folder exists, else make it
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # append to end
    with open(filename, mode='a') as file:
        writer = csv.writer(file)

        if rep == 1:
            writer.writerow(['Time', 'Total_Evaluations'])

        writer.writerow([time, total_evals])

def plot_avg(alg, bench_function, version, dims):
    path = build_path(alg, bench_function, version, dims)

    files = os.listdir(path)

    data = []

    for filename in files:
        if filename == 'time.csv':
            continue

        repetition = []

        with open(f'{path}/{filename}', mode='r') as file:
            reader = csv.DictReader(file)

            for row in reader:
                repetition.append(float(row["curbest"]))

        data.append(repetition)

    all_best_y = list(it.zip_longest(*data, fillvalue=np.nan))[:10000]
    mean_best_y = np.nanmean(all_best_y, axis=1)

    x = range(1, 10001)

    plt.semilogy(x, mean_best_y, label=alg.__name__)
    plt.fill_between(x, np.percentile(all_best_y, 5, axis=1), np.percentile(all_best_y, 95, axis=1), alpha=0.2)


if __name__ == '__main__':
    from ppa import PlantPropagation
    from fireworks import Fireworks
    import benchmarks

    dimension = 10
    plot_avg(PlantPropagation, benchmarks.sphere, "DEFAULT", dimension)
    plot_avg(Fireworks, benchmarks.sphere, "DEFAULT", dimension)

    plt.xlabel('Evaluation')
    plt.ylabel('Benchmark score')
    plt.title('Mean of bests')
    plt.legend()

    plt.show()
