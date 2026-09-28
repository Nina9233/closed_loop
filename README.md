# Closed-Loop Threat Response Simulation — README

A scoped-down **2D prototype** of a detect → estimate → decide → actuate loop for a host satellite
denying a non-cooperative docking attempt.

Dependencies: `pip install numpy matplotlib` (Python 3.9+). No GPU, no paid or licence-locked tools.

## Files
- `simulate.py` — the simulation: dynamics, sensor model, filter, decision logic, actuation, self-checks, Monte Carlo, sensitivity sweep
- `make_results.py` — runs the Monte Carlo (N=200) and 5 sensitivity sweeps (80 runs/point, Wilson confidence intervals) and the example trajectory → produces `results.png`, `example_trajectory.png`, `results_summary.txt`
- `paired_all.py` / `paired_latency.py` — paired comparisons: the same scenarios run under two settings of one parameter, compared with an exact McNemar test
- `extra_checks.py` — the perfect-sensor self-check, a latency test under late-detection conditions, and the metrics requested in the assignment brief (track initiation range, estimation error, delta-v, torque impulse, FOV-loss episodes); output saved in `extra_checks.log`
- `diag_arm.py` — diagnostic showing when, and at what true range, the host first arms, split by denied vs. docked runs; output saved in `diag_arm_output.txt`
- `plot_trajectory.py` — finds and plots one representative example run

## How to run

**A single scenario** (one randomised run, printed result — the quickest way to see the model work):
```python
import numpy as np
import simulate as sim

rng = np.random.default_rng(0)      # any seed
scenario = sim.sample_scenario(rng)
result = sim.simulate(scenario, rng)
print(result)
# {'outcome': 'docked' or 'denied_dv_exhausted' or 'timeout', 'denied': bool,
#  'time_s': ..., 'track_init_range': ..., 'dv_used': ..., ...}
```
Save this as e.g. `run_one.py` in the same folder and run `python run_one.py`; it completes in well under a second. Change the seed to see a different randomised approach.

**The full analysis** (self-checks, Monte Carlo, sweeps, plots, paired tests — everything behind the numbers in the design note):
```
python simulate.py --selfcheck     # self-checks, ~30 s
python make_results.py             # Monte Carlo (N=200) + all 5 sweeps + plots, ~6-7 min
python paired_all.py               # paired sensitivity tests, ~4 min
python diag_arm.py                 # arming diagnostic, ~1 min
python extra_checks.py             # perfect-sensor bound, late-detection latency test, metrics, ~4 min
```
Running all five in sequence reproduces every number and figure in the design note in well under the brief's 10-minute target (roughly 15 minutes combined; `make_results.py` alone, which produces the headline figure and all plots, takes ~6-7 minutes).

**Windows:** any code that starts a `multiprocessing.Pool` must sit inside `if __name__ == "__main__":`. All scripts here already do this.

## Key design choices
- Detection-probability curve is a logistic function solved exactly from the two anchor points in the brief (Pd(200 m)=0.95, Pd(800 m)=0.40): r50 = 727.4 m, k = 0.00558 (larger r50 = better sensor, i.e. detection out to longer range).
- Confidence intervals use the Wilson score method rather than the normal (Wald) approximation, since it is more reliable at these sample sizes and proportions.
- The range/bearing filter uses alpha-beta gains from the critically-damped rule (alpha = 0.05, beta = alpha²/(2−alpha)), chosen so the filter's range-rate error stays well below the true closing speeds being estimated.
- A track must be at least 25 frames (5 s) old before it is allowed to trigger arming, and the decision logic uses a 300 s urgency horizon.
- Sensitivity is assessed with paired comparisons (identical scenarios re-run under two settings) rather than by eye from overlapping confidence intervals on independent samples, since paired tests are far more sensitive to real effects.

## Results (N = 200; sweeps 80 runs/point)
- **Denial rate: 27/200 = 13.5% (95% Wilson CI 9.4–18.9%).** 24 denials are `denied_dv_exhausted`, 3 are `timeout`.
- Only 38/200 scenarios ever led to arming; track initiation happens at a median **true range of 3 m** (IQR 2–9 m). The chaser must start inside the fixed 60° camera FOV (~1 in 6 scenarios) to be seen early. **The denial rate is limited primarily by sensing geometry.**
- Paired tests (100 scenarios, extreme settings): decision threshold 0.3 vs 0.9 → 24 vs 2 denials (p<0.0001); max slew 0.5 vs 8°/s → 3 vs 18 (p=0.0001); **no detectable effect** from latency 0 vs 10 s (p=1.0), detection range r50 300 vs 1500 m (p=0.38), or camera update rate 1 vs 20 Hz (p=1.0). "No detectable effect" is not proof of no effect — with 100 paired scenarios, only effects of roughly ten percentage points or more would reliably show.
- The closed loop is demonstrably present: in all 38 armed runs, the host's own slew removed the chaser from the camera's field of view at least once (0.20 episodes/run overall).
- Estimation error, once a track is 5 s old: range RMS 0.22 m, range-rate RMS 0.077 m/s (medians across runs). False-alarm-triggered manoeuvres: 0.000/run. Chaser re-aim delta-v used: mean 0.165 m/s (of a 1.0 m/s budget). Cumulative torque impulse (an upper bound on wheel-momentum use, since no wheel-momentum model is implemented): mean 1.65, max 13.8 N·m·s.

## Self-checks
1. **Open-loop sanity — passed.** With an inert host, the chaser docks in 100% of runs.
2. **Perfect-sensor bound — passed, with a caveat.** With zero noise, zero false alarms, an effectively infinite detection range and zero latency, the denial rate is 18.5%, above the 13.5% baseline. This is a ceiling with respect to sensing quality only; other settings (e.g. a more aggressive arming threshold) can exceed it, since the fixed 60° field of view is not changed by this check.
3. **Dynamics check — partly done.** The planar CW propagator matches the closed-form solution for a purely radial offset to within numerical precision over 500 s. The brief's companion check (zero-torque attitude conservation) was not run, since the prototype uses a single body angle rather than a quaternion model.
4. **Latency injection — does not show the expected effect.** Adding 5–10 s of decision latency does not measurably reduce the denial rate in this prototype, including when detection is deliberately made late (r50 = 150 m or 80 m). In this model, the arming decision either happens hundreds of metres out or not at all, and the outcome is governed by field-of-view geometry, the arming threshold and the chaser's re-aim cost model — none of which are sensitive to a few seconds of added latency at this scenario length. This is reported here as an explicit finding and a limitation of the prototype's perception–action coupling, not glossed over.

## Limitations
The camera and docking-port axis share a single body angle, which is the model's largest simplification: it determines whether the chaser is ever visible early enough to matter, and it means the host's own tracking behaviour (needed to keep sensing the chaser) also points its vulnerable port at it. The chaser follows a fixed, non-adversarial guidance policy and is moved kinematically rather than by the CW dynamics (which are implemented and validated, but not used in the Monte Carlo runs); only its re-aiming manoeuvres are charged against its delta-v budget. The estimator is an alpha-beta filter, not a Kalman filter, with a simple (non-statistical) track-initiation gate. The actuator is an ideal single-axis torque/rate-limited source rather than a reaction-wheel array, and no wheel-momentum model is implemented. False alarms occur at 2% per camera frame; the lidar is sampled at the camera's 5 Hz rate rather than its own 2 Hz, and its range is only used when the camera also detects. The relative-motion model has not been validated against a nonlinear two-body propagation over the full 30-minute scenario window. Confidence intervals and paired tests are based on limited sample sizes (200 for the headline, 80–100 per comparison), and interaction effects between parameters were not tested. Scenario sampling excludes combinations of range and closing rate that cannot physically complete within 30 minutes, which narrows the tested envelope slightly.
