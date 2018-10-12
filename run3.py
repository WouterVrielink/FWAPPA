from ppa import PlantPropagation
from fireworks import Fireworks
import benchmarks

N = 5
d = 2

bounds = [(-5.12, 5.12) for _ in range(d)]
bench_function = benchmarks.rastigrin

bounds = [(-3, 3), (-2, 2)]
bench_function = benchmarks.six_hump_camel

max_iter = 100
m = 50
m_roof = 5
a = 0.04
b = 0.8
max_amp = 40

# (self, N, d, bounds, bench_function, max_iter, m, m_roof, a, b, max_amp)
alg = Fireworks(N, d, bounds, bench_function, max_iter, m, m_roof, a, b, max_amp)
alg.start()
