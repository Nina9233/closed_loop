"""Extra checks and the metrics required by the brief (Section 5 of the assignment)."""
import numpy as np
from multiprocessing import Pool
import simulate as sim
from paired_latency import mcnemar_exact

def run(args):
    seed, params = args
    rng = np.random.default_rng(seed); sc = sim.sample_scenario(rng)
    r = sim.simulate(sc, rng, params=params)
    return r

def denied(rs): return np.array([r["denied"] for r in rs])

if __name__ == "__main__":
    import sys
    METRICS_ONLY = "--metrics-only" in sys.argv
    N = 200
    with Pool() as pool:
        base = pool.map(run, [(s, {}) for s in range(N)])
        if METRICS_ONLY: perfect = base
        else: perfect = pool.map(run, [(s, {"noise_scale": 0.0, "p_fa": 0.0, "r50": 1e6, "latency_s": 0.0}) for s in range(N)])
        if not METRICS_ONLY: print(f"[perfect-sensor bound] baseline denial {denied(base).mean()*100:.1f}%  |  zero noise, zero false alarms, "
              f"'infinite' range, zero latency: {denied(perfect).mean()*100:.1f}%", flush=True)
        # latency with tight time margins (late detection): does added latency hurt the defence?
        for r50 in (727.4, 150.0, 80.0):
            for L in (0.0, 5.0):
                pass
        M = 200
        for r50 in (() if METRICS_ONLY else (150.0, 80.0)):
            a = pool.map(run, [(s, {"r50": r50, "latency_s": 0.0}) for s in range(M)])
            b = pool.map(run, [(s, {"r50": r50, "latency_s": 5.0}) for s in range(M)])
            da, db = denied(a), denied(b)
            x = int(((da == 1) & (db == 0)).sum()); y = int(((da == 0) & (db == 1)).sum())
            print(f"[latency, r50={r50:.0f} m] denied {da.sum()}/{M} (0 s) vs {db.sum()}/{M} (5 s) | flipped denied->docked {x}, docked->denied {y} | exact McNemar p={mcnemar_exact(x,y):.4f}", flush=True)
    # required metrics on the baseline MC
    def col(k): return [r[k] for r in base if r[k] is not None]
    print("\n=== Metrics on the baseline Monte Carlo (N=200) ===")
    tir = np.array(col("track_init_range")); print(f"runs with a confirmed track: {len(tir)}/{N}")
    print(f"track initiation TRUE range: median {np.median(tir):.0f} m (IQR {np.percentile(tir,25):.0f}-{np.percentile(tir,75):.0f})")
    rr = np.array(col("rms_range_err")); vr = np.array(col("rms_rate_err"))
    print(f"estimation error after maturity (per-run RMS): range median {np.median(rr):.2f} m, range-rate median {np.median(vr):.3f} m/s")
    fa = np.mean([r["false_alarm_manoeuvres"] for r in base]); print(f"false-alarm-triggered manoeuvres per run: {fa:.3f}")
    armed_runs = [r for r in base if r["first_arm"] is not None]
    print(f"runs in which the host armed: {len(armed_runs)}/{N}")
    print(f"FOV-loss episodes per run (all): {np.mean([r['sensor_loss_episodes'] for r in base]):.2f}; "
          f"caused while armed (own manoeuvre): {np.mean([r['fov_loss_while_armed'] for r in base]):.2f} per run, "
          f"{sum(r['fov_loss_while_armed']>0 for r in base)} runs with >=1")
    print(f"chaser delta-v used (re-aiming only): mean {np.mean([r['dv_used'] for r in base]):.3f} m/s, max {np.max([r['dv_used'] for r in base]):.3f} m/s (budget 1.0)")
    wm = np.array([r["wheel_momentum"] for r in base]); print(f"cumulative torque impulse (upper bound on wheel momentum use): mean {wm.mean():.2f}, max {wm.max():.2f} N*m*s")
    tt = [r["time_s"] for r in base if not r["denied"]]; print(f"time to dock (docked runs): median {np.median(tt):.0f} s")
    print("outcomes:", {k: sum(r['outcome']==k for r in base) for k in set(r['outcome'] for r in base)})
