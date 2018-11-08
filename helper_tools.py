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


def check_folder(filename):
    # Check if folder exists, else make it
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_to_csv(alg, filename):
    x, y = alg.env.get_evaluation_statistics()
    _, best_y = alg.env.get_evaluation_statistics_best()
    _, generations = alg.env.get_generation_statistics()

    check_folder(filename)

    with open(filename, mode='w') as file:
        writer = csv.writer(file)

        writer.writerow(['evaluation', 'value', 'curbest', 'generation'])

        for row in zip(x, y, best_y, generations):
            writer.writerow(row)


def save_time(time, total_evals, rep, filename):
    check_folder(filename)

    # append to end
    with open(filename, mode='a') as file:
        writer = csv.writer(file)

        if rep == 1:
            writer.writerow(['Time', 'Total_Evaluations'])

        writer.writerow([time, total_evals])


def get_data(path, file_list, column_name):
    data = []

    for filename in file_list:
        if filename == 'time.csv':
            continue

        repetition = []

        with open(f'{path}/{filename}', mode='r') as file:
            reader = csv.DictReader(file)

            for row in reader:
                repetition.append(float(row[column_name]))

        data.append(repetition)

    return data


def plot_avg(alg, bench_function, version, dim, correction=0):
    path = build_path(alg, bench_function, version, dim)

    file_list = os.listdir(path)

    data = get_data(path, file_list, "curbest")

    N = len(file_list) - 1

    all_best_y = np.matrix(list(it.zip_longest(*data, fillvalue=np.nan)))[:10000] - correction
    mean_best_y = np.nanmean(all_best_y, axis=1)

    x = range(1, 10001)

    plt.semilogy(x, mean_best_y, label=f'{alg.__name__} N={N}')
    plt.fill_between(x, np.percentile(all_best_y, 5, axis=1), np.percentile(all_best_y, 95, axis=1), alpha=0.2)


def plot_end(alg, bench_function, version):
    avgs = []
    err_lo = []
    err_hi = []

    for dim in range(2, 101):
        path = build_path(alg, bench_function, version, dim)

        file_list = os.listdir(path)

        data = get_data(path, file_list, "curbest")

        N = len(file_list) - 1

        all_best_y = list(it.zip_longest(*data, fillvalue=np.nan))[9999]
        avgs.append(np.nanmean(all_best_y))

        err_lo.append(np.percentile(all_best_y, 5))
        err_hi.append(np.percentile(all_best_y, 95))

    plt.errorbar(range(2, 101), avgs, yerr=[err_lo, err_hi], fmt='o', label=f'{alg.__name__} N={N}', capsize=2)


def get_plot_path(bench_function):
    return f'plots/versus/{bench_function.__name__}/'


def plot_versus(bench_function, dims, version="DEFAULT", correction=0):
    path = get_plot_path(bench_function)
    filename = f'{dims}d.png'

    check_folder(path)

    # Clear any existing figure
    plt.clf()

    plot_avg(PlantPropagation, bench_function, version, dims, correction=correction)
    plot_avg(Fireworks, bench_function, version, dims, correction=correction)

    plt.xlabel('Evaluation')
    plt.ylabel('Benchmark score')
    plt.title('Mean of bests')
    plt.legend()

    plt.savefig(f'{path}/{filename}', bbox_inches='tight')


def plot_versus_dims(bench_function, version="DEFAULT"):
    path = get_plot_path(bench_function)
    filename = f'all_dims.png'

    check_folder(path)

    # Clear any existing figure
    plt.clf()

    plot_end(PlantPropagation, bench_function, version)
    plot_end(Fireworks, bench_function, version)

    ax = plt.gca()
    ax.set_yscale("log", nonposy='clip')

    plt.xlabel('Dimension')
    plt.ylabel('Benchmark score')
    plt.title('Mean of results after 10000 evaluations')
    plt.legend()

    plt.savefig(f'{path}/{filename}', bbox_inches='tight')


if __name__ == '__main__':
    from ppa import PlantPropagation
    from fireworks import Fireworks
    import benchmarks

    # Branin: 0.397887
    # Easom: -1
    # Goldstein_price: 3
    # six_hump_camel: -1.0316

    print("Plotting 2d benchmarks...")

    # plot_versus(benchmarks.easom, 2, correction=-1)
    plot_versus(benchmarks.branin, 2, correction=0.397887)
    # plot_versus(benchmarks.goldstein_price, 2, correction=3)
    # plot_versus(benchmarks.martin_gaddy, 2)
    # plot_versus(benchmarks.six_hump_camel, 2, correction=-1.0316)

    # for dims in range(2, 101):
    #     print(f'Plotting Nd benchmarks {dims}d/100d...')
    #
    #     plot_versus(benchmarks.sphere, dims)
    #     plot_versus(benchmarks.tablet, dims)
    #     plot_versus(benchmarks.cigar, dims)
    #     plot_versus(benchmarks.elipse, dims)
    #     plot_versus(benchmarks.schwefel, dims)
    #     plot_versus(benchmarks.rastigrin, dims)
    #     plot_versus(benchmarks.sphere, dims)
    #     plot_versus(benchmarks.ackley, dims)
    #     plot_versus(benchmarks.rosenbrock, dims)

    # print("Plotting Nd benchmarks...")
    #
    # plot_versus_dims(benchmarks.sphere)
    # plot_versus_dims(benchmarks.tablet)
    # plot_versus_dims(benchmarks.cigar)
    # plot_versus_dims(benchmarks.elipse)
    # plot_versus_dims(benchmarks.schwefel)
    # plot_versus_dims(benchmarks.rastigrin)
    # plot_versus_dims(benchmarks.sphere)
    # plot_versus_dims(benchmarks.ackley)
    # plot_versus_dims(benchmarks.rosenbrock)
