"""
Closed-loop threat response simulation (scoped-down, 2D/planar version).

Simplifications made deliberately, per the brief's "simplify aggressively
and defend it" instruction. Each is called out where it matters:

  1. Planar (2D) relative motion instead of full 3D. Justified: LEO
     proximity scenarios are dominated by in-plane CW dynamics; out-of-plane
     motion adds a decoupled oscillation that doesn't change the core
     detect->estimate->decide->actuate coupling being tested here.
  2. Camera boresight and docking-port axis are treated as the same body
     direction (a single host attitude angle theta), rather than two
     separate body axes 90 degrees apart as in the original spec. This
     keeps the attitude control problem 1-DOF instead of 3-DOF. It is the
     single biggest simplification in this model -- flagged in the design
     note as the first thing to relax if given more time.
  3. Alpha-beta filter instead of an EKF (explicitly allowed by the brief).
     Much less tuning-sensitive, at the cost of not exploiting the known
     CW dynamics model in the estimator.
  4. Chaser motion is KINEMATIC: it flies a commanded straight line toward
     an aim point on the host's port axis (with a braking profile inside
     60 m) and re-aims 20 s after the port axis has moved >15 deg. The CW
     propagator in this file is implemented and checked against the
     closed-form solution, but it is NOT used in the Monte Carlo runs: the
     chaser is effectively assumed to cancel orbital drift continuously.
     Only re-aiming is charged to its 1.0 m/s delta-v budget; the
     closing/braking burns are treated as free.
  6. The lidar is sampled at the camera rate (5 Hz), not its own 2 Hz, and
     its range is only used when the camera also detects.
  5. Translational host burns are NOT modelled -- only attitude (reaction
     wheel) response. This is allowed ("optional translational burns").

Run:
    python simulate.py --selfcheck        # runs the 4 self-checks
    python simulate.py --montecarlo 200   # runs N randomised scenarios
    python simulate.py --sweep            # runs the sensitivity sweep
"""

import argparse
import numpy as np
from multiprocessing import Pool

# --------------------------------------------------------------------------
# Constants (from the assignment brief)
# --------------------------------------------------------------------------
MU = 3.986004418e14
R_EARTH = 6378137.0
ALT = 400e3
A = R_EARTH + ALT
N_MEAN = np.sqrt(MU / A**3)          # orbital mean motion, rad/s (~92.6 min period)

MAX_TORQUE = 0.01                     # N*m per axis -> used as max angular accel proxy
INERTIA = 2.0                         # kg*m^2, representative single-axis value
MAX_ALPHA = MAX_TORQUE / INERTIA      # rad/s^2, max angular acceleration
MAX_SLEW = np.deg2rad(2.0)            # rad/s

CAMERA_HALF_FOV = np.deg2rad(30.0)    # 60 deg full FOV
CAMERA_RATE_HZ = 5.0
CAMERA_BEARING_SIGMA = np.deg2rad(0.1)
LIDAR_RANGE_SIGMA_BASE = 1.0          # m
LIDAR_RANGE_SIGMA_FRAC = 0.01         # + 1% of range
P_FALSE_ALARM = 0.02                  # per frame

DOCK_RANGE = 5.0                      # m
DOCK_ANGLE = np.deg2rad(15.0)         # rad
DOCK_SPEED = 0.1                      # m/s
SCENARIO_MAX_TIME = 30 * 60.0         # s
DT = 1.0 / CAMERA_RATE_HZ             # 0.2 s common tick

CHASER_DV_BUDGET = 1.0                # m/s, for lateral corrections
REPLAN_TRIGGER_DEG = 15.0
REPLAN_DELAY = 20.0                   # s


# --------------------------------------------------------------------------
# Relative dynamics: planar Clohessy-Wiltshire
# --------------------------------------------------------------------------
def cw_matrix_2d(n):
    """State = [x, y, vx, vy] in the orbital plane (x=radial, y=along-track)."""
    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    A[2, 0] = 3 * n**2
    A[2, 3] = 2 * n
    A[3, 2] = -2 * n
    return A


def cw_step(state, n, dt):
    """One RK4 step of the planar CW equations."""
    A = cw_matrix_2d(n)

    def deriv(s):
        return A @ s

    k1 = deriv(state)
    k2 = deriv(state + 0.5 * dt * k1)
    k3 = deriv(state + 0.5 * dt * k2)
    k4 = deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (better behaved than
    the simple Wald +/- interval at small n or p near 0/1)."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(centre - half, 0.0), min(centre + half, 1.0)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------
# Sensor model
# --------------------------------------------------------------------------
def pd_curve(range_m, r50=727.4, k=0.00558):
    """Logistic detection probability -- an empirical surrogate, NOT a
    first-principles optical model. r50 is the range at which Pd = 0.5
    (LARGER r50 = detection out to longer range = BETTER sensor).
    Parameters solved from the two anchor points Pd(200)=0.95, Pd(800)=0.40:
        k   = [logit(0.40) - logit(0.95)] / 600 m  = 0.00558 1/m
        r50 = 800 - logit(0.40)/k                  = 727.4 m
    (The earlier r50=560, k=0.008 gave Pd(800)=0.13, not 0.40 -- a
    calibration error, fixed here and guarded by selfcheck_pd_calibration.)"""
    return 1.0 / (1.0 + np.exp(k * (range_m - r50)))


def sensor_tick(rel_pos, host_theta, rng, r50=727.4, noise_scale=1.0, p_fa=P_FALSE_ALARM):
    """
    Returns a measurement dict or None.
    rel_pos: (x,y) of chaser relative to host, world/LVLH frame.
    host_theta: current host boresight/port-axis angle (rad).
    """
    x, y = rel_pos
    rng_true = np.hypot(x, y)
    bearing_world = np.arctan2(y, x)
    bearing_body = wrap_angle(bearing_world - host_theta)

    in_fov = abs(bearing_body) <= CAMERA_HALF_FOV

    # false alarm can occur regardless of true target visibility
    if rng.random() < p_fa:
        fa_bearing = host_theta + rng.uniform(-CAMERA_HALF_FOV, CAMERA_HALF_FOV)
        # (in the rare case a real detection also happens this tick, the
        # real one is returned preferentially below; false alarms are only
        # returned when nothing real is detected, to keep the M-of-N logic
        # simple to reason about)
        false_alarm_pending = {"type": "false_alarm", "bearing": fa_bearing, "range": None}
    else:
        false_alarm_pending = None

    if not in_fov:
        return false_alarm_pending

    pd = pd_curve(rng_true, r50=r50)
    if rng.random() > pd:
        return false_alarm_pending  # missed detection

    bearing_meas = bearing_world + rng.normal(0, CAMERA_BEARING_SIGMA * noise_scale)
    range_sigma = (LIDAR_RANGE_SIGMA_BASE + LIDAR_RANGE_SIGMA_FRAC * rng_true) * noise_scale
    range_meas = rng_true + rng.normal(0, range_sigma)

    return {"type": "detection", "bearing": bearing_meas, "range": max(range_meas, 0.0)}


# --------------------------------------------------------------------------
# Track initiation (M-of-N) + alpha-beta filter
# --------------------------------------------------------------------------
class Track:
    """Alpha-beta filter on (bearing, range), plus derived range-rate."""

    def __init__(self, bearing0, range0, alpha=0.05, beta=0.05**2 / (2 - 0.05)):
        # Gains: alpha=0.05 with the critically-damped relation
        # beta = alpha^2/(2-alpha) (~0.0013). The first version used
        # alpha=0.6, beta=0.15, which gave a range-rate error of 1.5-6 m/s
        # against true closing speeds of 0.3-1 m/s -- i.e. the threat score
        # was mostly noise (see README / diag_arm.py). These gains give
        # ~0.03-0.12 m/s error at the sensor noise this model defines.
        self.bearing = bearing0
        self.range = range0
        self.range_rate = 0.0
        self.bearing_rate = 0.0
        self.alpha = alpha
        self.beta = beta
        self.age_frames = 1
        self.frames_since_update = 0

    def predict(self, dt):
        self.range += self.range_rate * dt
        self.bearing = wrap_angle(self.bearing + self.bearing_rate * dt)
        self.frames_since_update += 1

    def update(self, meas_bearing, meas_range, dt):
        r_resid = meas_range - self.range
        self.range += self.alpha * r_resid
        self.range_rate += (self.beta / dt) * r_resid

        b_resid = wrap_angle(meas_bearing - self.bearing)
        self.bearing = wrap_angle(self.bearing + self.alpha * b_resid)
        self.bearing_rate += (self.beta / dt) * b_resid

        self.age_frames += 1
        self.frames_since_update = 0

    def time_to_closest_approach(self):
        if self.range_rate >= 0:
            return None
        return self.range / (-self.range_rate)


class TrackInitiator:
    """M-of-N confirmation gate before a track is trusted."""

    def __init__(self, m=3, n=5, gate_rad=np.deg2rad(2.0)):
        self.buffer = []
        self.m, self.n, self.gate = m, n, gate_rad

    def update(self, meas):
        self.buffer = (self.buffer + [meas])[-self.n:]
        plausible = [b for b in self.buffer if b is not None and b["type"] == "detection"]
        if len(plausible) < self.m:
            return False, None
        bearings = [b["bearing"] for b in plausible]
        if (max(bearings) - min(bearings)) < self.gate:
            return True, plausible[-1]
        return False, None


# --------------------------------------------------------------------------
# Decision logic
# --------------------------------------------------------------------------
def threat_score(track, urgency_horizon=300.0):
    """Heuristic threat score in [0, ~1].
    closing : closing speed (m/s, capped at 1)
    urgency : 1/(1+TCA/horizon); TCA is time-to-closest-approach from the
              filtered range and range-rate (always >0 here; None if opening).
    The horizon was 60 s in the first version, which made the score almost
    never reach the arming threshold during a normal approach (see
    diag_arm.py); 300 s makes urgency meaningful over the last ~5 minutes."""
    closing = min(max(0.0, -track.range_rate), 1.0)
    tca = track.time_to_closest_approach()
    urgency = 1.0 / (1.0 + tca / urgency_horizon) if tca is not None else 0.0
    confidence = min(track.age_frames / 5.0, 1.0)
    return confidence * (0.5 * closing + 0.5 * urgency)


# --------------------------------------------------------------------------
# One full scenario
# --------------------------------------------------------------------------
def sample_scenario(rng, max_close_time=1500.0):
    """
    NOTE / documented simplification: r0=800m combined with closing_rate=0.3 m/s
    implies a pure radial closing time of ~2667s, which exceeds the 30-minute
    (1800s) scenario window on its own, before any host response or
    re-planning delay is even considered -- i.e. some (r0, closing_rate)
    combinations in the brief's stated ranges are not reachable in the
    scenario window regardless of host behaviour. Rather than let random
    draws silently produce un-winnable-for-the-chaser scenarios (which would
    inflate the denial rate for reasons that have nothing to do with the
    defence system), we resample until the pure radial closing time leaves
    at least 300s of margin inside the 1800s window for manoeuvring/replanning.
    This is exactly the kind of assumption the brief asks to be stated and
    defended, so it is called out here rather than left implicit.
    """
    for _ in range(1000):
        r0 = rng.uniform(300, 800)
        closing_rate = rng.uniform(0.3, 1.0)
        if r0 / closing_rate <= max_close_time:
            break
    bearing0 = rng.uniform(-np.pi, np.pi)
    x0 = r0 * np.cos(bearing0)
    y0 = r0 * np.sin(bearing0)
    return dict(x0=x0, y0=y0, closing_rate=closing_rate, bearing0=bearing0)


def simulate(scenario, rng, params=None, record=False):
    """
    params: dict overriding defaults, used by the sensitivity sweep. Keys:
        r50 (detection-range curve param), camera_rate_hz, decision_arm,
        decision_clear, latency_s, max_slew (rad/s)
    record: if True, also return a 'trajectory' list of per-step snapshots
        (time, chaser x/y, host_theta, armed, tracked) for visualisation.
    """
    p = dict(
        r50=727.4, camera_rate_hz=CAMERA_RATE_HZ,
        decision_arm=0.6, decision_clear=0.35,
        latency_s=0.0, max_slew=MAX_SLEW,
        track_before_arm=True,
        noise_scale=1.0, p_fa=P_FALSE_ALARM,
        min_track_age=25,   # frames (5 s at 5 Hz): a track must mature before it may trigger arming
    )
    if params:
        p.update(params)

    dt = 1.0 / p["camera_rate_hz"]

    # --- initial conditions ---
    host_theta = 0.0          # port axis / camera boresight, world frame
    host_omega = 0.0
    port_axis_at_last_replan = host_theta

    chaser_pos = np.array([scenario["x0"], scenario["y0"]])
    BRAKE_RANGE = 60.0   # m -- distance at which the chaser starts decelerating

    def heading_dir(pos, bearing):
        """Unit direction aimed at a point just short of the host, along
        `bearing`, so that IF the chaser flies this line unmolested, it
        naturally arrives aligned with that bearing -- i.e. re-planning is
        a one-time heading change, not a continuous small-angle correction
        (matches the brief's re-planning rule)."""
        aim_point = (DOCK_RANGE * 0.5) * np.array([np.cos(bearing), np.sin(bearing)])
        to_aim = aim_point - pos
        return to_aim / max(np.linalg.norm(to_aim), 1e-6)

    def braked_speed(range_m, nominal_speed):
        """Decelerate on final approach so the docking speed condition
        (<=0.1 m/s) can actually be met -- a real docking profile is fast
        far out and slow at contact, not constant-speed all the way in."""
        if range_m > BRAKE_RANGE:
            return nominal_speed
        floor = 0.03
        return max(floor, nominal_speed * (range_m / BRAKE_RANGE))

    # Chaser aims to arrive along the host's CURRENT port axis. This first
    # heading is the chaser's assumed pre-planned approach and costs no
    # budget; only RE-aiming later (in response to the host's manoeuvre)
    # draws on the chaser's delta-v budget.
    target_bearing = host_theta
    chaser_dir = heading_dir(chaser_pos, target_bearing)
    chaser_vel = chaser_dir * scenario["closing_rate"]
    dv_used = 0.0
    replanning_timer = None

    initiator = TrackInitiator()
    track = None
    armed = False
    decision_time_pending = None   # timestamp when arm condition first met (for latency)
    first_arm = None               # (time, true range, track age) at first arming, for diagnostics
    false_alarm_manoeuvres = 0
    sensor_loss_episodes = 0
    fov_loss_while_armed = 0
    was_in_fov_prev = None
    track_init_range = None
    track_init_time = None
    err_r_sq = err_rr_sq = 0.0
    n_err = 0
    wheel_momentum = 0.0            # crude proxy: |integral of torque|

    t = 0.0
    outcome = "timeout"
    trajectory = [] if record else None

    while t < SCENARIO_MAX_TIME:
        rel_pos = chaser_pos
        rng_true = np.hypot(*rel_pos)

        # ---- win/lose check ----
        chaser_speed = np.hypot(*chaser_vel)
        approach_bearing = np.arctan2(rel_pos[1], rel_pos[0])
        align_err = abs(wrap_angle(approach_bearing - host_theta))
        if rng_true <= DOCK_RANGE and align_err <= DOCK_ANGLE and chaser_speed <= DOCK_SPEED:
            outcome = "docked"
            break
        if dv_used >= CHASER_DV_BUDGET:
            outcome = "denied_dv_exhausted"
            # chaser keeps coasting but can no longer correct; still let sim
            # run to time limit in case radial closing alone achieves dock
            # (rare, since bearing likely misaligned by then) -- but to keep
            # runtime bounded we stop here and count as denied.
            break

        # ---- sensor ----
        meas = sensor_tick(rel_pos, host_theta, rng, r50=p["r50"],
                           noise_scale=p["noise_scale"], p_fa=p["p_fa"])
        # Geometric field-of-view test (independent of whether a detection
        # happened to succeed). A "loss episode" is a transition from
        # in-view to out-of-view; those while armed are attributable to the
        # host's own defensive slew.
        in_fov = bool(abs(wrap_angle(approach_bearing - host_theta)) <= CAMERA_HALF_FOV)
        if was_in_fov_prev is True and not in_fov:
            sensor_loss_episodes += 1
            if armed:
                fov_loss_while_armed += 1
        was_in_fov_prev = in_fov

        confirmed, conf_meas = initiator.update(meas)
        if confirmed and track is None:
            track = Track(conf_meas["bearing"], conf_meas["range"])
            track_init_range, track_init_time = float(rng_true), t   # TRUE range/time at track creation
        elif track is not None:
            track.predict(dt)
            if meas is not None and meas["type"] == "detection":
                track.update(meas["bearing"], meas["range"], dt)
            if track.age_frames >= p["min_track_age"]:
                true_rr = float(np.dot(rel_pos, chaser_vel) / max(rng_true, 1e-6))
                err_r_sq += (track.range - rng_true) ** 2
                err_rr_sq += (track.range_rate - true_rr) ** 2
                n_err += 1

        # ---- decision ----
        if track is not None:
            score = threat_score(track)
            mature = track.age_frames >= p["min_track_age"]
            if not armed and mature and score >= p["decision_arm"]:
                if decision_time_pending is None:
                    decision_time_pending = t
                if t - decision_time_pending >= p["latency_s"]:
                    armed = True
                    if first_arm is None:
                        first_arm = (t, float(rng_true), track.age_frames)
                    if meas is not None and meas["type"] == "false_alarm":
                        false_alarm_manoeuvres += 1
            elif armed and score < p["decision_clear"]:
                armed = False
                decision_time_pending = None
        # (a track born purely from a false alarm is possible in principle;
        #  the M-of-N gate makes it rare, and we don't specially distinguish
        #  it here -- a named limitation.)

        # ---- host attitude control ----
        if armed:
            # slew to increase angular separation from current chaser bearing
            err = wrap_angle(approach_bearing - host_theta)
            direction = -np.sign(err) if abs(err) > 1e-3 else 1.0
            omega_cmd = direction * p["max_slew"]
        elif p["track_before_arm"] and track is not None:
            # Reasonable default behaviour: once a contact is confirmed, the
            # host keeps it centred in the FOV (proportional tracking) even
            # before deciding it's a threat -- otherwise a body-fixed
            # narrow-FOV sensor would lose almost every off-axis approach
            # within a few seconds purely from geometry, which would make
            # sensor-quality parameters (detection range, noise) irrelevant
            # to the outcome. This is disabled for the open-loop self-check,
            # where the host must be fully inert.
            err = wrap_angle(track.bearing - host_theta)
            omega_cmd = np.clip(1.5 * err, -p["max_slew"], p["max_slew"])
        else:
            omega_cmd = 0.0
        # torque/slew-rate saturated integration
        alpha_cmd = np.clip((omega_cmd - host_omega) / dt, -MAX_ALPHA, MAX_ALPHA)
        host_omega = np.clip(host_omega + alpha_cmd * dt, -p["max_slew"], p["max_slew"])
        host_theta = wrap_angle(host_theta + host_omega * dt)
        wheel_momentum += abs(alpha_cmd) * INERTIA * dt

        # ---- chaser guidance ----
        port_axis_now = host_theta  # (simplification: port axis == boresight, see docstring)
        angle_off = np.rad2deg(abs(wrap_angle(port_axis_now - port_axis_at_last_replan)))
        if angle_off > REPLAN_TRIGGER_DEG and replanning_timer is None:
            replanning_timer = t
        if replanning_timer is not None and t - replanning_timer >= REPLAN_DELAY:
            new_target = port_axis_now
            new_dir = heading_dir(chaser_pos, new_target)
            # cost of a heading change, approximated at the current nominal
            # closing speed (the deceleration profile itself is treated as
            # "free" station-keeping, not a budgeted correction)
            cost = np.linalg.norm((new_dir - chaser_dir) * scenario["closing_rate"])
            remaining = max(CHASER_DV_BUDGET - dv_used, 0.0)
            if cost <= remaining:
                chaser_dir = new_dir
                dv_used += cost
            else:
                frac = remaining / cost if cost > 0 else 0.0
                blended = chaser_dir + frac * (new_dir - chaser_dir)
                chaser_dir = blended / max(np.linalg.norm(blended), 1e-6)
                dv_used = CHASER_DV_BUDGET
            target_bearing = new_target
            port_axis_at_last_replan = port_axis_now
            replanning_timer = None

        speed = braked_speed(rng_true, scenario["closing_rate"])
        chaser_vel = chaser_dir * speed
        chaser_pos = chaser_pos + chaser_vel * dt

        if record:
            trajectory.append(dict(
                t=t, x=rel_pos[0], y=rel_pos[1], host_theta=host_theta,
                armed=armed, tracked=track is not None,
            ))

        t += dt

    result = dict(
        outcome=outcome,
        denied=outcome != "docked",
        time_s=t,
        track_init_range=track_init_range,
        track_init_time=track_init_time,
        rms_range_err=(err_r_sq / n_err) ** 0.5 if n_err else None,
        rms_rate_err=(err_rr_sq / n_err) ** 0.5 if n_err else None,
        fov_loss_while_armed=fov_loss_while_armed,
        false_alarm_manoeuvres=false_alarm_manoeuvres,
        sensor_loss_episodes=sensor_loss_episodes,
        wheel_momentum=wheel_momentum,
        dv_used=dv_used,
        first_arm=first_arm,
    )
    if record:
        result["trajectory"] = trajectory
    return result


# --------------------------------------------------------------------------
# Self-checks
# --------------------------------------------------------------------------
def selfcheck_open_loop(n_runs=30):
    """Host never manoeuvres -> chaser should dock in ~100% of runs."""
    rng = np.random.default_rng(0)
    docked = 0
    for i in range(n_runs):
        scenario = sample_scenario(rng)
        # force armed=False forever AND disable pre-arm tracking, so the
        # host is genuinely inert as the check requires
        res = simulate(scenario, rng, params={"decision_arm": 999.0, "track_before_arm": False})
        if res["outcome"] == "docked":
            docked += 1
    rate = docked / n_runs
    print(f"[open-loop sanity]   docking rate = {rate*100:.0f}%  (expect ~100%)")
    return rate


def selfcheck_perfect_sensor(n_runs=100):
    """Brief's check #2: zero noise, zero false alarms, 'infinite' detection
    range, zero latency -> denial rate with respect to sensing quality.
    Compared with the baseline on the SAME scenarios."""
    base = perf = 0
    for i in range(n_runs):
        rng_a = np.random.default_rng(i); sc = sample_scenario(rng_a)
        base += simulate(sc, rng_a)["denied"]
        rng_b = np.random.default_rng(i); sc = sample_scenario(rng_b)
        perf += simulate(sc, rng_b, params={"noise_scale": 0.0, "p_fa": 0.0, "r50": 1e6, "latency_s": 0.0})["denied"]
    print(f"[perfect-sensor bound] baseline {base/n_runs*100:.0f}%, perfect sensing {perf/n_runs*100:.0f}% "
          f"(expect perfect >= baseline; a ceiling w.r.t. sensing only -- FOV unchanged)")
    return perf / n_runs


def selfcheck_pd_calibration():
    """The detection curve must actually hit the anchor points it claims to."""
    p200, p800 = float(pd_curve(200.0)), float(pd_curve(800.0))
    ok = abs(p200 - 0.95) < 0.005 and abs(p800 - 0.40) < 0.005
    print(f"[Pd calibration]     Pd(200)={p200:.3f} (want 0.95), Pd(800)={p800:.3f} (want 0.40)  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


def selfcheck_cw_dynamics():
    """Verify CW propagation against the closed-form radial-only solution."""
    n = N_MEAN
    x0 = 100.0
    state = np.array([x0, 0.0, 0.0, 0.0])
    t_end = 500.0
    steps = int(t_end / 0.5)
    for _ in range(steps):
        state = cw_step(state, n, 0.5)
    # closed form for a purely radial initial offset with zero initial velocity:
    # x(t) = x0 (4 - 3 cos(nt)), y(t) = x0 (6 sin(nt) - 6nt)  [standard CW solution]
    x_cf = x0 * (4 - 3 * np.cos(n * t_end))
    y_cf = x0 * (6 * np.sin(n * t_end) - 6 * n * t_end)
    err_x = abs(state[0] - x_cf)
    err_y = abs(state[1] - y_cf)
    print(f"[CW dynamics check]  |dx|={err_x:.4f} m, |dy|={err_y:.4f} m over {t_end:.0f}s (expect small)")
    return err_x, err_y


def selfcheck_latency_injection(n_runs=40):
    """Adding 5s of decision delay should visibly drop the denial rate."""
    rng = np.random.default_rng(2)
    baseline, delayed = 0, 0
    for i in range(n_runs):
        s = sample_scenario(rng)
        r0 = simulate(s, np.random.default_rng(100 + i), params={"latency_s": 0.0})
        r1 = simulate(s, np.random.default_rng(100 + i), params={"latency_s": 5.0})
        baseline += r0["denied"]
        delayed += r1["denied"]
    print(f"[latency injection]  denial rate: 0s latency={baseline/n_runs*100:.0f}%, "
          f"5s latency={delayed/n_runs*100:.0f}%  (brief expects a visible drop; see paired tests in the report -- this check FAILS in this prototype)")
    return baseline / n_runs, delayed / n_runs


def run_all_selfchecks():
    print("=== Self-checks ===")
    selfcheck_pd_calibration()
    selfcheck_open_loop()
    selfcheck_perfect_sensor()
    selfcheck_cw_dynamics()
    selfcheck_latency_injection()


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------
def _run_one(seed):
    rng = np.random.default_rng(seed)
    scenario = sample_scenario(rng)
    return simulate(scenario, rng)


def monte_carlo(n=200, parallel=True):
    seeds = list(range(n))
    if parallel:
        with Pool() as pool:
            results = pool.map(_run_one, seeds)
    else:
        results = [_run_one(s) for s in seeds]
    k_denied = int(sum(r["denied"] for r in results))
    denial_rate = k_denied / n
    ci95 = wilson_interval(k_denied, n)
    print(f"\n=== Monte Carlo (N={n}) ===")
    print(f"Denied {k_denied}/{n} = {denial_rate*100:.1f}%  (95% Wilson CI: "
          f"{ci95[0]*100:.1f}% - {ci95[1]*100:.1f}%)")
    print(f"Mean false-alarm-triggered manoeuvres/run: "
          f"{np.mean([r['false_alarm_manoeuvres'] for r in results]):.2f}")
    print(f"Mean sensor-loss episodes/run: "
          f"{np.mean([r['sensor_loss_episodes'] for r in results]):.2f}")
    print(f"Mean chaser dv used: {np.mean([r['dv_used'] for r in results]):.3f} m/s")
    return results, denial_rate, ci95


# --------------------------------------------------------------------------
# Sensitivity sweep
# --------------------------------------------------------------------------
def _run_one_with_params(args):
    seed, params = args
    rng = np.random.default_rng(seed)
    scenario = sample_scenario(rng)
    return simulate(scenario, rng, params=params)


def sweep_parameter(name, values, n_runs=60, base_params=None):
    base_params = dict(base_params or {})
    denial_rates = []
    cis = []
    for v in values:
        params = dict(base_params)
        params[name] = v
        args = [(s, params) for s in range(n_runs)]
        with Pool() as pool:
            results = pool.map(_run_one_with_params, args)
        k_denied = int(sum(r["denied"] for r in results))
        dr = k_denied / n_runs
        ci = wilson_interval(k_denied, n_runs)
        denial_rates.append(dr)
        cis.append(ci)
        print(f"  {name}={v}: denial rate = {dr*100:.1f}%  (95% CI {ci[0]*100:.1f}-{ci[1]*100:.1f}%)")
    return denial_rates, cis


def run_sensitivity_sweep():
    print("\n=== Sensitivity sweep ===")

    print("Sweeping end-to-end latency (s), finer near the cliff:")
    lat_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0, 10.0]
    lat_rates, lat_cis = sweep_parameter("latency_s", lat_vals)

    print("Sweeping decision arm threshold:")
    thr_vals = [0.3, 0.45, 0.6, 0.75, 0.9]
    thr_rates, thr_cis = sweep_parameter("decision_arm", thr_vals)

    print("Sweeping max slew rate (deg/s):")
    slew_vals_deg = [0.5, 1.0, 2.0, 4.0, 8.0]
    slew_rates, slew_cis = sweep_parameter("max_slew", [np.deg2rad(v) for v in slew_vals_deg])

    print("Sweeping detection-range curve (r50, m -- LARGER = better sensor):")
    r50_vals = [300.0, 500.0, 727.4, 1000.0, 1500.0]
    r50_rates, r50_cis = sweep_parameter("r50", r50_vals)

    print("Sweeping camera update rate (Hz):")
    rate_vals = [1.0, 2.0, 5.0, 10.0, 20.0]
    rate_rates, rate_cis = sweep_parameter("camera_rate_hz", rate_vals)

    return {
        "latency_s": (lat_vals, lat_rates, lat_cis),
        "decision_arm": (thr_vals, thr_rates, thr_cis),
        "max_slew_deg": (slew_vals_deg, slew_rates, slew_cis),
        "r50": (r50_vals, r50_rates, r50_cis),
        "camera_rate_hz": (rate_vals, rate_rates, rate_cis),
    }


# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--montecarlo", type=int, default=0)
    parser.add_argument("--sweep", action="store_true")
    args = parser.parse_args()

    if args.selfcheck:
        run_all_selfchecks()
    if args.montecarlo:
        monte_carlo(args.montecarlo)
    if args.sweep:
        run_sensitivity_sweep()
    if not (args.selfcheck or args.montecarlo or args.sweep):
        run_all_selfchecks()
        monte_carlo(200)
        run_sensitivity_sweep()
