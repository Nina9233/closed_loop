import numpy as np
from multiprocessing import Pool
import simulate as sim
def run(args):
    seed, lat = args
    rng = np.random.default_rng(seed); sc = sim.sample_scenario(rng)
    r = sim.simulate(sc, rng, params={"latency_s": lat})
    return r["denied"], r["first_arm"], float(np.hypot(sc["x0"], sc["y0"]))
if __name__ == "__main__":
    with Pool() as pool:
        for lat in (0.0, 1.0):
            out = pool.map(run, [(s, lat) for s in range(200)])
            for label, flag in (("DENIED", True), ("DOCKED", False)):
                g = [o for o in out if o[0] == flag and o[1] is not None]
                if not g: print(f"lat={lat} {label}: n=0"); continue
                ar = np.array([o[1][2] for o in g]); rg = np.array([o[1][1] for o in g]); r0 = np.array([o[2] for o in g])
                print(f"lat={lat} {label}: n={len(g)} | track age at first arm: median {np.median(ar):.0f} frames | true range at arm: median {np.median(rg):.0f} m (initial range median {np.median(r0):.0f} m)")
