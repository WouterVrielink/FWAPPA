from ppa import PlantPropagation
import benchmarks
import time

N = 2
start = time.clock()
for i in range(1, 101):
    loop = time.clock()

    d = i
    bounds = [(-100, 100) for _ in range(d)]
    bench_function = benchmarks.sphere
    # bounds = [(-2, 2), (-1, 1)]
    # bench_function = benchmarks.six_hump_camel
    max_iter = 10000
    max_runners = 5
    m = N

    alg = PlantPropagation(N, d, bounds, bench_function, max_iter, max_runners, m)
    alg.start()

    loop2 = time.clock()

    print("Dimension {} took {} seconds. Total time {}".format(i, loop2 - loop, loop2 - start))
