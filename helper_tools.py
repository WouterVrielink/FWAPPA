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

    plt.semilogy(x, mean_best_y, label=f'{alg.__name__} (N={N})')
    plt.fill_between(x, np.percentile(all_best_y, 5, axis=1), np.percentile(all_best_y, 95, axis=1), alpha=0.2)


def plot_end_all_dims(alg, bench_function, version):
    avgs = []
    err_lo = []
    err_hi = []

    for dim in range(2, 101):
        path = build_path(alg, bench_function, version, dim)

        file_list = os.listdir(path)

        data = get_data(path, file_list, "curbest")

        N = len(file_list) - 1

        all_best_y = list(it.zip_longest(*data, fillvalue=np.nan))[9999]
        avg = np.nanmean(all_best_y)

        avgs.append(avg)

        # We need the absolute errors
        err_lo.append(avg - np.percentile(all_best_y, 5))
        err_hi.append(np.percentile(all_best_y, 95) - avg)

    plt.errorbar(range(2, 101), avgs, yerr=[err_lo, err_hi], fmt='o', label=f'{alg.__name__} (N={N})', capsize=2)


def plot_end_all_shifts(alg, bench_function, shifts, version, correction=0):
    avgs = []
    err_lo = []
    err_hi = []

    for value in shifts:
        if value != 0:
            bench_function_add, _ = benchmarks.apply_add(bench_function, [(0, 0), (0, 0)], value=value)
        else:
            bench_function_add = bench_function

        path = build_path(alg, bench_function_add, version, 2)

        file_list = os.listdir(path)

        data = get_data(path, file_list, "curbest")

        N = len(file_list) - 1

        all_best_y = np.matrix(list(it.zip_longest(*data, fillvalue=np.nan))[9999]) - correction
        avg = np.nanmean(all_best_y)

        avgs.append(avg)

        # We need the absolute errors
        err_lo.append(avg - np.percentile(all_best_y, 5))
        err_hi.append(np.percentile(all_best_y, 95) - avg)

    plt.errorbar(shifts, avgs, yerr=[err_lo, err_hi], fmt='o', label=f'{alg.__name__} (N={N})', capsize=2)


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

    plot_end_all_dims(PlantPropagation, bench_function, version)
    plot_end_all_dims(Fireworks, bench_function, version)

    ax = plt.gca()
    ax.set_yscale("log", nonposy='clip')

    plt.xlabel('Dimension')
    plt.ylabel('Benchmark score')
    plt.title('Mean of results after 10000 evaluations')
    plt.legend()

    plt.savefig(f'{path}/{filename}', bbox_inches='tight')


def plot_versus_shift(bench_function, shifts, version="DEFAULT", correction=0):
    path = get_plot_path(bench_function)
    filename = f'shifts.png'

    check_folder(path)

    # Clear any existing figure
    plt.clf()

    plot_end_all_shifts(PlantPropagation, bench_function, shifts, version, correction=correction)
    plot_end_all_shifts(Fireworks, bench_function, shifts, version, correction=correction)

    ax = plt.gca()
    ax.set_yscale('log', nonposy='clip')
    ax.set_xscale('symlog', linthreshx=0.1)

    plt.xlabel('Shift')
    plt.ylabel('Benchmark score')
    plt.title('Mean of results after 10000 evaluations')
    plt.legend()

    plt.savefig(f'{path}/{filename}', bbox_inches='tight')


def plot_compare_center_single(bench_function, bench_function_center, version="DEFAULT", correction=0):
    path = get_plot_path(bench_function)
    filename = f'centered.png'

    check_folder(path)

    # Clear any existing figure
    plt.clf()

    plot_avg(PlantPropagation, bench_function, version, 2, correction=correction)
    plot_avg(PlantPropagation, bench_function_center, version, 2, correction=correction)

    plot_avg(Fireworks, bench_function, version, 2, correction=correction)
    plot_avg(Fireworks, bench_function_center, version, 2, correction=correction)

    ax = plt.gca()
    ax.set_yscale('symlog', nonposy='clip', linthreshy=0.0000001)

    plt.xlabel('Shift')
    plt.ylabel('Benchmark score')
    plt.title('Mean of results after 10000 evaluations')
    plt.legend()

    plt.savefig(f'{path}/{filename}', bbox_inches='tight')


if __name__ == '__main__':
    from ppa import PlantPropagation
    from fireworks import Fireworks
    import benchmarks

    print("Plotting 2d benchmarks...")

    # Comparison between non-centered function and the centered version
    for bench_function in benchmarks.two_dim_non_centered_bench_functions():
        bounds, correction = benchmarks.two_dim_bench_functions()[bench_function]
        bench_function_center, _ = benchmarks.apply_add(bench_function, bounds, name='_center')

        plot_compare_center_single(bench_function, bench_function_center, correction=correction)

    # Comparison between fwa and ppa, and comparison for different shift sizes
    for bench_function, (domain, correction) in benchmarks.two_dim_bench_functions().items():
        plot_versus(bench_function, 2, correction=correction)

        plot_versus_shift(bench_function, (0, 0.1, 1, 10, 100, 1000), correction=correction)

    # Comparisons between fwa and ppa for both unshifted and shifted benchmarks per dimension
    for dims in range(2, 101):
        print(f'Plotting Nd benchmarks {dims}d/100d...')

        for bench_function, domain in benchmarks.n_dim_bench_functions().items():
            plot_versus(bench_function, dims)

            domain = [domain for _ in range(dims)]
            bench_function_add, domain_add = benchmarks.apply_add(bench_function, domain)

            plot_versus(bench_function_add, dims)

    print("Plotting Nd benchmark commparisons...")

    # Comparisons over all dimensions for shifted and unshifted benchmarks
    for bench_function, domain in benchmarks.n_dim_bench_functions().items():
        plot_versus_dims(bench_function)

        domain = [domain for _ in range(100)]
        bench_function_add, domain_add = benchmarks.apply_add(bench_function, domain)

        plot_versus_dims(bench_function_add)

    # TODO: Comparison of N-D shifted vs unshifted?
    # - Use results at 10k evals of all dims
    # - Seperate FWA and PPA so we only have 2 lines
