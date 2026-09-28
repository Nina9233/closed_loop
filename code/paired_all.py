"""Paired endpoint comparisons (same scenarios, N=100) for each swept parameter."""
import numpy as np
from multiprocessing import Pool
import simulate as sim
from paired_latency import mcnemar_exact

def run(args):
    seed, params = args
    rng = np.random.default_rng(seed)
    sc = sim.sample_scenario(rng)
    return sim.simulate(sc, rng, params=params)["denied"]

CASES = [
    ("latency 0 vs 10 s", {"latency_s": 0.0}, {"latency_s": 10.0}),
    ("decision_arm 0.3 vs 0.9", {"decision_arm": 0.3}, {"decision_arm": 0.9}),
    ("max_slew 0.5 vs 8 deg/s", {"max_slew": np.deg2rad(0.5)}, {"max_slew": np.deg2rad(8.0)}),
    ("r50 300 vs 1500 m", {"r50": 300.0}, {"r50": 1500.0}),
    ("camera 1 vs 20 Hz", {"camera_rate_hz": 1.0}, {"camera_rate_hz": 20.0}),
]

if __name__ == "__main__":
    N = 100
    with Pool() as pool:
        for name, pa, pb in CASES:
            a = np.array(pool.map(run, [(s, pa) for s in range(N)]))
            b = np.array(pool.map(run, [(s, pb) for s in range(N)]))
            x = int(((a == 1) & (b == 0)).sum()); y = int(((a == 0) & (b == 1)).sum())
            print(f"{name:26s}: {a.sum():>2}/{N} vs {b.sum():>2}/{N} | A-only denied {x}, B-only denied {y} | exact McNemar p={mcnemar_exact(x, y):.4f}", flush=True)
