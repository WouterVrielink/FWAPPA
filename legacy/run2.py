from ppa import PlantPropagation
from fireworks import Fireworks
import benchmarks
import helper_tools

import numpy as np
import matplotlib.pyplot as plt
import itertools as it

N = 30
d = 2
# bounds = [(-5.12, 5.12) for _ in range(d)]
# bench_function = benchmarks.rastigrin
# bounds = [(-6.12, 4.12) for _ in range(d)]
# bench_function = benchmarks.rastigrin_add
# bounds = [(-3, 3), (-2, 2)]
# bench_function = benchmarks.six_hump_camel
bounds = [(-100, 100) for _ in range(d)]
bench_function = benchmarks.sphere

max_iter = 100
max_runners = 5
m = N

reps = 10

# Fireworks
# max_iter = 100
# m = 50
# m_roof = 5
# a = 0.04
# b = 0.8
# max_amp = 40


all_best_y = []
for repetition in range(reps):
    print("Rep: ", repetition)
    # alg = Fireworks(N, d, bounds, bench_function, max_iter, m, m_roof, a, b, max_amp)

    alg = PlantPropagation(N, d, bounds, bench_function, max_iter, max_runners, m)
    alg.start()

    _, best_y = alg.env.get_evaluation_statistics_best()

    all_best_y.append(best_y)

    # helper_tools.save_to_csv(x, y, best_x, best_y, alg, bench_function, repetition, 'evaluations')

all_best_y = list(it.zip_longest(*all_best_y, fillvalue=np.nan))
mean_best_y = np.nanmean(all_best_y, axis=1)
std_best_y = np.nanstd(all_best_y, axis=1)

plt.semilogy(range(1, len(mean_best_y) + 1), mean_best_y, label=r'$\frac{1}{2}*(\tanh{(4*f(x) - 2)} + 1)$')
plt.fill_between(range(1, len(mean_best_y) + 1), np.nanmin(all_best_y, axis=1), np.nanmax(all_best_y, axis=1), alpha=0.2)

# bounds = [(-110, 90) for _ in range(d)]
# bench_function = benchmarks.sphere_add

# max_iter = 100
# m = 50
# m_roof = 5
# a = 0.04
# b = 0.8
# max_amp = 40

all_best_y = []
for repetition in range(reps):
    print("Rep: ", repetition)
    # (self, N, d, bounds, bench_function, max_iter, m, m_roof, a, b, max_amp)
    # alg = Fireworks(N, d, bounds, bench_function, max_iter, m, m_roof, a, b, max_amp)
    alg = PlantPropagation(N, d, bounds, bench_function, max_iter, max_runners, m, tanh_mod=2)

    alg.start()

    _, best_y = alg.env.get_evaluation_statistics_best()

    all_best_y.append(best_y)

all_best_y = np.array(list(it.zip_longest(*all_best_y, fillvalue=np.nan)))
mean_best_y = np.nanmean(all_best_y, axis=1)
# std = np.nanstd(all_best_y, axis=1)
# std_plus = mean_best_y + std
# std_min = mean_best_y - std
# std_min = np.where(mean_best_y - std < mean_best_y[-1], mean_best_y[-1], mean_best_y - std)

plt.semilogy(range(1, len(mean_best_y) + 1), mean_best_y, label=r'$\frac{1}{2}*(\tanh{(8*f(x) - 4)} + 1)$')
plt.fill_between(range(1, len(mean_best_y) + 1), np.nanmin(all_best_y, axis=1), np.nanmax(all_best_y, axis=1), alpha=0.2)
plt.xlabel('Evaluation')
plt.ylabel('Benchmark score')
plt.title('PPA mean of bests')
plt.legend()
plt.show()
