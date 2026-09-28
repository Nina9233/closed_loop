"""Paired comparison: identical 200 scenarios (same seeds) run at each latency.
Exact McNemar (sign) test on discordant pairs."""
import numpy as np
from multiprocessing import Pool
from math import comb
import simulate as sim

def run(args):
    seed, lat = args
    rng = np.random.default_rng(seed)
    sc = sim.sample_scenario(rng)
    return sim.simulate(sc, rng, params={"latency_s": lat})["denied"]

def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(p, 1.0)

if __name__ == "__main__":
    N = 200
    lats = [0.0, 0.2, 0.5, 1.0]
    with Pool() as pool:
        res = {L: np.array(pool.map(run, [(s, L) for s in range(N)])) for L in lats}
    base = res[0.0]
    print(f"Baseline (latency 0): denied {base.sum()}/{N} = {base.mean()*100:.1f}%")
    for L in lats[1:]:
        x = res[L]
        b = int(((base == 1) & (x == 0)).sum())   # denied at 0s, docked at L
        c = int(((base == 0) & (x == 1)).sum())   # docked at 0s, denied at L
        print(f"latency {L:>3}s: denied {x.sum()}/{N} = {x.mean()*100:.1f}% | "
              f"scenarios flipped denied->docked: {b}, docked->denied: {c} | "
              f"exact McNemar p = {mcnemar_exact(b, c):.4f}")
