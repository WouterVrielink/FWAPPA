import csv
import os
import itertools as it
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sc


def build_path(alg, bench_function, version, dims, prefix=None):
    if prefix:
        return f'data/{alg.__name__}_{version}/{prefix}/{bench_function.__name__}/{dims}d'
    return f'data/{alg.__name__}_{version}/{bench_function.__name__}/{dims}d'


def get_name(alg, bench_function, version, dims, repetition, prefix=None):
    return f'{build_path(alg, bench_function, version, dims, prefix)}/{str(repetition)}.csv'


def get_time_name(alg, bench_function, version, dims, prefix=None):
    return f'{build_path(alg, bench_function, version, dims, prefix)}/time.csv'


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
    # TODO filename to filepath?

    # append to end
    with open(filename, mode='a') as file:
        writer = csv.writer(file)

        if rep == 1:
            writer.writerow(['Time', 'Total_Evaluations'])

        writer.writerow([time, total_evals])


def save_data(filepath, data, header):
    check_folder(filepath)

    with open(filepath, mode='w') as file:
        writer = csv.writer(file)

        writer.writerow(header)

        for i, row in enumerate(data):
            writer.writerow([i, row])


def get_data(path, file_list, column_name):
    data = []

    for filename in file_list:
        if filename == 'time.csv' or filename == 'compact.csv':
            continue

        repetition = []

        with open(f'{path}/{filename}', mode='r') as file:
            reader = csv.DictReader(file)

            for row in reader:
                repetition.append(float(row[column_name]))

        data.append(repetition)

    return data


def compact_data(path):
    # TODO move file_list to get_data?
    file_list = os.listdir(path)

    data = get_data(path, file_list, 'curbest')

    new_data = [sub[9999] for sub in data]

    save_data(f'{path}/compact.csv', new_data, ['Repetition', 'Value@10k'])


def get_color(alg):
    # Blue if PPA, orange if FWA
    return '#1f77b4' if alg == PlantPropagation else '#ff7f0e'


def plot_median(alg, bench_function, version, dim, correction=0):
    path = build_path(alg, bench_function, version, dim)

    file_list = os.listdir(path)

    data = get_data(path, file_list, "curbest")

    all_best_y = np.matrix(list(it.zip_longest(*data, fillvalue=np.nan)))[:10000] - correction
    median_best_y = np.median(all_best_y, axis=1)

    x = range(1, 10001)

    color = get_color(alg)

    plt.semilogy(x, median_best_y, label=f'{alg.__name__}', color=color)
    plt.fill_between(x, np.percentile(all_best_y, 0, axis=1), np.percentile(all_best_y, 25, axis=1), alpha=0.2, color=color, linewidth=0.0)
    plt.fill_between(x, np.percentile(all_best_y, 25, axis=1), np.percentile(all_best_y, 75, axis=1), alpha=0.4, color=color, linewidth=0.0)
    plt.fill_between(x, np.percentile(all_best_y, 75, axis=1), np.percentile(all_best_y, 100, axis=1), alpha=0.2, color=color, linewidth=0.0)


def wilcoxon_test(alg, bench_function, bench_function_add, version="DEFAULT", dim="2"):
    path = build_path(alg, bench_function, version, dim)
    path_add = build_path(alg, bench_function_add, version, dim)

    file_list = os.listdir(path)
    file_list_add = os.listdir(path)

    data = get_data(path, file_list, "curbest")
    data_add = get_data(path_add, file_list_add, "curbest")

    print(f'{alg.__name__}, {bench_function.__name__}')

    data = np.matrix(list(it.zip_longest(*data, fillvalue=np.nan)))[9999]
    data_add = np.matrix(list(it.zip_longest(*data_add, fillvalue=np.nan)))[9999]

    data = np.squeeze(data).ravel().tolist()[0]
    data_add = np.squeeze(data_add).ravel().tolist()[0]

    print(sc.wilcoxon(data, data_add))
    print(sc.ranksums(data, data_add))
    print(sc.mannwhitneyu(data, data_add))


def plot_end_all_dims(alg, bench_function, version):
    medians = []
    err_lo = []
    err_hi = []

    for dim in range(2, 101):
        path = build_path(alg, bench_function, version, dim)
        filename = 'compact.csv'
        column_name = 'Value@10k'

        # TODO move this to a function (as well as in get_data)
        data = []
        with open(f'{path}/{filename}', mode='r') as file:
            reader = csv.DictReader(file)

            for row in reader:
                data.append(float(row[column_name]))

        median = np.median(data)

        medians.append(median)

        # We need the absolute errors, not the "height" of the values
        err_lo.append(median - np.percentile(data, 0))
        err_hi.append(np.percentile(data, 100) - median)

    plt.errorbar(range(2, 101), medians, yerr=[err_lo, err_hi], fmt='o', label=f'{alg.__name__}', capsize=2, color=get_color(alg))


def plot_times(version="DEFAULT"):
    fpath = 'plots/times/'
    filename = f'times.png'

    check_folder(fpath)

    plt.clf()

    for alg in (Fireworks, PlantPropagation):
        mean = []
        std = []

        xs = []
        ys = []

        for dims in range(2, 101):
            temp = []

            for bench_function, _ in benchmarks.n_dim_bench_functions().items():
                path = get_time_name(alg, bench_function, version, dims)
                with open(path, mode='r') as file:
                    reader = csv.DictReader(file)

                    for row in reader:
                        temp.append(float(row["Time"]))

            xs += [dims]*len(temp)
            ys += temp

            mean.append(np.mean(temp))
            std.append(np.std(temp))

        plt.errorbar(range(2, 101), mean, yerr=[std, std], fmt='o', label=f'{alg.__name__}', capsize=2, color=get_color(alg))

        # Fit the corresponding lines
        slope, intercept, r_value, p_value, std_err = sc.linregress(xs, ys)
        print(slope, intercept, r_value, p_value, std_err)

    ax = plt.gca()
    ax.set_xlim((2, 101))
    plt.xlabel('Dimension')
    plt.ylabel('Time (in seconds)')
    plt.title('Time to complete 10000 evaluations')
    plt.legend()

    plt.savefig(f'{fpath}/{filename}', bbox_inches='tight')


def plot_end_all_shifts(alg, bench_function, shifts, version, correction=0):
    medians = []
    err_lo = []
    err_hi = []

    for value in shifts:
        if value != 0:
            bench_function_add = benchmarks.apply_add(bench_function, value=value)
        else:
            bench_function_add = bench_function

        path = build_path(alg, bench_function_add, version, 2)

        file_list = os.listdir(path)

        data = get_data(path, file_list, "curbest")

        all_best_y = np.matrix(list(it.zip_longest(*data, fillvalue=np.nan))[9999]) - correction
        median = np.percentile(all_best_y, 50)

        medians.append(median)

        # We need the absolute errors
        err_lo.append(median - np.percentile(all_best_y, 0))
        err_hi.append(np.percentile(all_best_y, 100) - median)

    plt.errorbar(shifts, medians, yerr=[err_lo, err_hi], fmt='o', label=f'{alg.__name__}', capsize=2, color=get_color(alg))


def get_plot_path(bench_function):
    return f'plots/versus/{bench_function.__name__}/'


def plot_versus(bench_function, dims, version="DEFAULT", correction=0, title=False):
    path = get_plot_path(bench_function)
    filename = f'{bench_function.__name__}_{dims}d.png'

    check_folder(path)

    # Clear any existing figure
    plt.clf()

    plot_median(Fireworks, bench_function, version, dims, correction=correction)
    plot_median(PlantPropagation, bench_function, version, dims, correction=correction)

    plt.xlabel('Evaluation')
    plt.ylabel('Objective value (normalised)')

    if title:
        plt.title(f'Benchmark results (N=10, {bench_function.official_name})')
    else:
        plt.title(f'{bench_function.official_name} Function', fontsize=14, fontweight='bold')

    plt.legend()

    plt.savefig(f'{path}/{filename}', bbox_inches='tight')


def plot_versus_dims(bench_function, version="DEFAULT", title=False):
    path = get_plot_path(bench_function)
    filename = f'{bench_function.__name__}_all_dims.png'

    check_folder(path)

    # Clear any existing figure
    plt.clf()

    plot_end_all_dims(Fireworks, bench_function, version)
    plot_end_all_dims(PlantPropagation, bench_function, version)

    ax = plt.gca()
    ax.set_yscale("log", nonposy='clip')

    ax.set_xlim((2, 101))

    plt.xlabel('Dimension')
    plt.ylabel('Objective value (normalised)')

    if title:
        plt.title(f'Results after 10000 evaluations (N=10, {bench_function.official_name})')
    else:
        plt.title(f'{bench_function.official_name} Function (shifted)', fontsize=14, fontweight='bold')

    plt.legend(loc='lower right')

    plt.savefig(f'{path}/{filename}', bbox_inches='tight')


def plot_versus_shift(bench_function, shifts, version="DEFAULT", correction=0, title=False):
    path = get_plot_path(bench_function)
    filename = f'{bench_function.__name__}_shifts.png'

    check_folder(path)

    # Clear any existing figure
    plt.clf()

    plot_end_all_shifts(Fireworks, bench_function, shifts, version, correction=correction)
    plot_end_all_shifts(PlantPropagation, bench_function, shifts, version, correction=correction)

    ax = plt.gca()
    ax.set_yscale('log', nonposy='clip')
    ax.set_xscale('symlog', linthreshx=0.1)

    plt.xlabel('Amount of shift')
    plt.ylabel('Objective value (normalized)')

    if title:
        plt.title(f'Results after 10000 evaluations (N=10, {bench_function.official_name})')
    else:
        plt.title(f'{bench_function.official_name} Function', fontsize=14, fontweight='bold')

    plt.legend(loc='lower right')

    plt.savefig(f'{path}/{filename}', bbox_inches='tight')


def plot_compare_center_single(bench_function, bench_function_center, version="DEFAULT", correction=0, title=False):
    path = get_plot_path(bench_function)
    filename = f'{bench_function.__name__}_centered.png'

    check_folder(path)

    # Clear any existing figure
    plt.clf()

    plot_median(Fireworks, bench_function, version, 2, correction=correction)
    plot_median(Fireworks, bench_function_center, version, 2, correction=correction)

    plot_median(PlantPropagation, bench_function, version, 2, correction=correction)
    plot_median(PlantPropagation, bench_function_center, version, 2, correction=correction)

    ax = plt.gca()
    ax.set_yscale('symlog', nonposy='clip', linthreshy=0.0000001)

    plt.xlabel('Amount of shift')
    plt.ylabel('Objective value (normalized)')

    if title:
        plt.title(f'Results after 10000 evaluations (N=10, {bench_function.official_name})')
    else:
        plt.title(f'{bench_function.official_name} Function', fontsize=14, fontweight='bold')

    plt.legend()

    plt.savefig(f'{path}/{filename}', bbox_inches='tight')


if __name__ == '__main__':
    from ppa import PlantPropagation
    from fireworks import Fireworks
    import benchmarks

    bench_fun = [getattr(benchmarks, fun) for fun in dir(benchmarks) if hasattr(getattr(benchmarks, fun), 'is_n_dimensional')]
    two_dim_fun = [fun for fun in bench_fun if not fun.is_n_dimensional]
    n_dim_fun = [fun for fun in bench_fun if fun.is_n_dimensional]

    non_center_two_dim_fun = [fun for fun in two_dim_fun if (0, 0) not in fun.global_minima]
    non_center_n_dim_fun = [fun for fun in n_dim_fun if (0) not in fun._global_minima]

    algorithms = (Fireworks, PlantPropagation)

    # # Compact data from 21x21 grid search
    # grid_number = 21
    # for alg in algorithms:
    #     for bench in bench_fun:
    #         print(bench.__name__)
    #         for x_i in range(grid_number):
    #             for y_i in range(grid_number):
    #                 compact_data(f'data/{alg.__name__}_DEFAULT/shifted_domain/{bench.__name__}_{x_i}_{y_i}/2d')
    #                 compact_data(f'data/{alg.__name__}_DEFAULT/unshifted_domain/{bench.__name__}_{x_i}_{y_i}/2d')

    # # Compact data from N-dimensional tests
    # for alg in algorithms:
    #     for bench in n_dim_fun:
    #         print(bench.__name__)
    #
    #         for dims in range(2, 101):
    #
    #             bench.dims = dims
    #
    #             bench_add = benchmarks.apply_add(bench)
    #
    #             compact_data(build_path(alg, bench, 'DEFAULT', dims))
    #             compact_data(build_path(alg, bench_add, 'DEFAULT', dims))

    # # Plot computation times
    # plot_times()
    #
    # print("Plotting 2d benchmarks...")
    #
    # Comparison between non-centered function and the centered version
    for bench in non_center_two_dim_fun:
        bench_center = benchmarks.apply_add(bench, value=bench.global_minima[0], name='_center')

        plot_compare_center_single(bench, bench_center, correction=bench.correction)

    # TODO bench.correction kan naar de functie

    # Comparison between fwa and ppa, centered and non-centered, and comparison for different shift sizes
    for bench in two_dim_fun:
        plot_versus(bench, 2, correction=bench.correction)

        bench_center = benchmarks.apply_add(bench, value=bench.global_minima[0], name='_center')

        plot_versus(bench_center, 2, correction=bench_center.correction)
        plot_versus_shift(bench_center, (0, 0.1, 1, 10, 100, 1000), correction=bench_center.correction)
    #
    #     # Similarity statistics
    #     for alg in algorithms:
    #         wilcoxon_test(alg, bench, bench_center)
    #
    # # Comparisons between fwa and ppa for both unshifted and shifted benchmarks per dimension
    # for dims in range(2, 101):
    #     print(f'Plotting Nd benchmarks {dims}d/100d...')
    #
    #     for bench in n_dim_fun:
    #         bench.dims = dims
    #
    #         plot_versus(bench, dims)
    #
    #         bench_add = benchmarks.apply_add(bench)
    #
    #         plot_versus(bench_add, dims)

    # print("Plotting Nd benchmark commparisons...")
    # Comparisons over all dimensions for shifted and unshifted benchmarks
    # for bench in n_dim_fun:
    #     bench.dims = 2
    #
    #     plot_versus_dims(bench)
    #
    #     bench_add = benchmarks.apply_add(bench)
    #
    #     plot_versus_dims(bench_add)
