"""
Plots one representative example run: the chaser's path relative to the
host, with the camera field-of-view and docking cone shown, coloured by
whether the host has a track and whether it's actively evading. This is a
single illustrative case, not a statistical result -- see results.png /
results_summary.txt for the actual Monte Carlo and sweep data.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import simulate as sim


def find_example_seed(want_outcome="denied", start=0, max_tries=500):
    """Search for a seed that produces an interesting, illustrative run:
    one where a track was actually established and the host armed at
    least once, rather than a trivial always-in-FOV or never-detected case."""
    for seed in range(start, start + max_tries):
        rng = np.random.default_rng(seed)
        scenario = sim.sample_scenario(rng)
        res = sim.simulate(scenario, rng, record=True)
        traj = res["trajectory"]
        was_armed = any(pt["armed"] for pt in traj)
        was_tracked = any(pt["tracked"] for pt in traj)
        outcome_ok = (want_outcome == "any") or (res["outcome"] == want_outcome) or \
                     (want_outcome == "denied" and res["denied"])
        if was_armed and was_tracked and outcome_ok:
            return seed, scenario, res
    return None, None, None


def plot_example(seed, scenario, res, fname="example_trajectory.png"):
    traj = res["trajectory"]
    xs = np.array([p["x"] for p in traj])
    ys = np.array([p["y"] for p in traj])
    armed = np.array([p["armed"] for p in traj])
    tracked = np.array([p["tracked"] for p in traj])

    fig, ax = plt.subplots(figsize=(7, 7))

    # phase segments: not-tracked (grey), tracked-not-armed (blue), armed (red)
    def phase_of(i):
        if armed[i]:
            return "armed"
        if tracked[i]:
            return "tracked"
        return "unseen"

    colors = {"unseen": "#999999", "tracked": "#2b4a6f", "armed": "#c0392b"}
    labels_done = set()
    for i in range(len(xs) - 1):
        ph = phase_of(i)
        lbl = None
        if ph not in labels_done:
            lbl = {"unseen": "No track yet", "tracked": "Tracked (not armed)",
                   "armed": "Armed / evading"}[ph]
            labels_done.add(ph)
        ax.plot(xs[i:i+2], ys[i:i+2], color=colors[ph], linewidth=2, label=lbl)

    # host at origin
    ax.plot(0, 0, marker="s", color="black", markersize=10, zorder=5, label="Host")

    # camera FOV / port-axis wedge at start and end
    def draw_wedge(theta, radius, color, alpha, label=None):
        wedge = mpatches.Wedge((0, 0), radius,
                                np.rad2deg(theta - sim.CAMERA_HALF_FOV),
                                np.rad2deg(theta + sim.CAMERA_HALF_FOV),
                                color=color, alpha=alpha, label=label)
        ax.add_patch(wedge)

    r_max = max(np.hypot(xs, ys)) * 1.05
    draw_wedge(traj[0]["host_theta"], r_max, "#2b4a6f", 0.08, "Camera FOV (start)")
    draw_wedge(traj[-1]["host_theta"], r_max, "#c0392b", 0.10, "Camera FOV / port axis (end)")

    # docking cone near host
    dock_wedge = mpatches.Wedge((0, 0), 25.0,
                                 np.rad2deg(traj[-1]["host_theta"] - sim.DOCK_ANGLE),
                                 np.rad2deg(traj[-1]["host_theta"] + sim.DOCK_ANGLE),
                                 color="green", alpha=0.25, label="Docking cone (final)")
    ax.add_patch(dock_wedge)

    ax.plot(xs[0], ys[0], marker="o", color="black", markersize=6, zorder=5)
    ax.annotate("start", (xs[0], ys[0]), textcoords="offset points", xytext=(8, 8))
    ax.plot(xs[-1], ys[-1], marker="X", color="black", markersize=9, zorder=5)
    ax.annotate("end", (xs[-1], ys[-1]), textcoords="offset points", xytext=(8, 8))

    ax.set_aspect("equal")
    ax.set_xlabel("Radial, x (m)")
    ax.set_ylabel("Along-track, y (m)")
    ax.set_title(f"Example run (seed={seed}) -- outcome: {res['outcome']}\n"
                 f"closing_rate={scenario['closing_rate']:.2f} m/s, "
                 f"r0={np.hypot(scenario['x0'], scenario['y0']):.0f} m")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved {fname}  (outcome={res['outcome']}, seed={seed})")


def main():
    seed, scenario, res = find_example_seed(want_outcome="denied")
    if seed is None:
        print("No suitable 'denied' example found in range; trying any outcome...")
        seed, scenario, res = find_example_seed(want_outcome="any")
    if seed is not None:
        plot_example(seed, scenario, res)
    else:
        print("Could not find any run with both tracking and arming -- try widening the search.")


if __name__ == "__main__":
    main()
