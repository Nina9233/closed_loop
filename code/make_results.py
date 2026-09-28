import numpy as np
import matplotlib.pyplot as plt
import simulate as sim
import plot_trajectory


def yerr(rates, cis):
    r = np.array(rates)
    lo = np.array([c[0] for c in cis])
    hi = np.array([c[1] for c in cis])
    return [(r - lo) * 100, (hi - r) * 100]


def main():
    plt.rcParams.update({"font.size": 9.5})

    # ---------------------------------------------------------------------
    # 1. Monte Carlo denial rate with CI
    # ---------------------------------------------------------------------
    results, denial_rate, ci95 = sim.monte_carlo(200)

    # ---------------------------------------------------------------------
    # 2. Sensitivity sweeps -- all 5 parameters named in the brief, each
    #    with its own 95% confidence interval (not just the headline MC)
    # ---------------------------------------------------------------------
    lat_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0, 10.0]
    lat_rates, lat_cis = sim.sweep_parameter("latency_s", lat_vals, n_runs=80)

    thr_vals = [0.3, 0.45, 0.6, 0.75, 0.9]
    thr_rates, thr_cis = sim.sweep_parameter("decision_arm", thr_vals, n_runs=80)

    slew_vals_deg = [0.5, 1.0, 2.0, 4.0, 8.0]
    slew_rates, slew_cis = sim.sweep_parameter(
        "max_slew", [np.deg2rad(v) for v in slew_vals_deg], n_runs=80)

    r50_vals = [300.0, 500.0, 727.4, 1000.0, 1500.0]
    r50_rates, r50_cis = sim.sweep_parameter("r50", r50_vals, n_runs=80)

    rate_vals = [1.0, 2.0, 5.0, 10.0, 20.0]
    rate_rates, rate_cis = sim.sweep_parameter("camera_rate_hz", rate_vals, n_runs=80)

    # ---------------------------------------------------------------------
    # Figure: 2x3 panel (headline + 5 sweeps, all with error bars)
    # ---------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    ax = axes[0, 0]
    ax.bar(["Denial rate"], [denial_rate * 100],
           yerr=[[(denial_rate - ci95[0]) * 100], [(ci95[1] - denial_rate) * 100]],
           capsize=8, color="#2b4a6f", width=0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Denial rate (%)")
    ax.set_title(f"A. Monte Carlo outcome (N=200)\n{denial_rate*100:.1f}%  (95% Wilson CI {ci95[0]*100:.1f}-{ci95[1]*100:.1f}%)")

    ax = axes[0, 1]
    ax.errorbar(lat_vals, np.array(lat_rates) * 100, yerr=yerr(lat_rates, lat_cis),
                fmt="o-", color="#c0392b", linewidth=2, capsize=3)
    ax.set_xlabel("End-to-end decision latency (s)")
    ax.set_ylabel("Denial rate (%)")
    ax.set_title("B. Decision latency")
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.errorbar(thr_vals, np.array(thr_rates) * 100, yerr=yerr(thr_rates, thr_cis),
                fmt="o-", color="#2b4a6f", linewidth=2, capsize=3)
    ax.set_xlabel("Decision arm threshold")
    ax.set_ylabel("Denial rate (%)")
    ax.set_title("C. Decision threshold")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.errorbar(slew_vals_deg, np.array(slew_rates) * 100, yerr=yerr(slew_rates, slew_cis),
                fmt="o-", color="#5b8a4e", linewidth=2, capsize=3)
    ax.set_xlabel("Max slew rate (deg/s)")
    ax.set_ylabel("Denial rate (%)")
    ax.set_title("D. Max slew rate")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.errorbar(r50_vals, np.array(r50_rates) * 100, yerr=yerr(r50_rates, r50_cis),
                fmt="o-", color="#8e44ad", linewidth=2, capsize=3)
    ax.set_xlabel("Detection-range r50 (m, larger = better sensor)")
    ax.set_ylabel("Denial rate (%)")
    ax.set_title("E. Sensor detection range")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.errorbar(rate_vals, np.array(rate_rates) * 100, yerr=yerr(rate_rates, rate_cis),
                fmt="o-", color="#d68910", linewidth=2, capsize=3)
    ax.set_xlabel("Camera update rate (Hz)")
    ax.set_ylabel("Denial rate (%)")
    ax.set_title("F. Camera update rate")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("\nSaved results.png (all 5 swept parameters, with error bars)")

    # ---------------------------------------------------------------------
    # 3. Example trajectory plot
    # ---------------------------------------------------------------------
    seed, scenario, res = plot_trajectory.find_example_seed(want_outcome="denied")
    if seed is None:
        seed, scenario, res = plot_trajectory.find_example_seed(want_outcome="any")
    if seed is not None:
        plot_trajectory.plot_example(seed, scenario, res)

    # ---------------------------------------------------------------------
    # Save raw numbers
    # ---------------------------------------------------------------------
    with open("results_summary.txt", "w") as f:
        f.write(f"Monte Carlo (N=200): denial rate = {denial_rate*100:.1f}% (95% Wilson CI {ci95[0]*100:.1f}-{ci95[1]*100:.1f}%)\n\n")

        def write_sweep(name, vals, rates, cis):
            f.write(f"{name} sweep:\n")
            for v, r, c in zip(vals, rates, cis):
                f.write(f"  {v}: {r*100:.1f}%  (95% CI {c[0]*100:.1f}-{c[1]*100:.1f}%)\n")
            f.write("\n")

        write_sweep("Latency (s)", lat_vals, lat_rates, lat_cis)
        write_sweep("Decision threshold", thr_vals, thr_rates, thr_cis)
        write_sweep("Max slew (deg/s)", slew_vals_deg, slew_rates, slew_cis)
        write_sweep("Detection range r50 (m)", r50_vals, r50_rates, r50_cis)
        write_sweep("Camera update rate (Hz)", rate_vals, rate_rates, rate_cis)
    print("Saved results_summary.txt")


if __name__ == "__main__":
    main()
