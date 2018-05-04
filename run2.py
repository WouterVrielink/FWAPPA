from ppa import PlantPropagation
import benchmarks

N = 30
d = 2
bounds = [(-5.12, 5.12) for _ in range(d)]
bench_function = benchmarks.rastigrin
# bounds = [(-3, 3), (-2, 2)]
# bench_function = benchmarks.six_hump_camel
max_iter = 10
max_runners = 5
m = N

alg = PlantPropagation(N, d, bounds, bench_function, max_iter, max_runners, m)
alg.start()
