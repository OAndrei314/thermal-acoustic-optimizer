# thermal-acoustic-optimizer

*Maintained by: claude-actions-daily-routine · Status: Active*
A from-scratch simulation + local-search optimizer for a fan-speed control curve, trading
off power consumption and acoustic noise against a hard safety temperature limit — modeled
on real thermal/fan calibration work (power and acoustic optimization on telecom hardware),
fully synthetic and simplified, not a reproduction of any real chassis or dataset.

## Research + money thesis

**Research question:** starting from an always-on baseline and a naive linear ramp, how
much power and acoustic noise can a simple local-search optimizer save on a fixed
synthetic workload, without ever crossing the safety temperature limit?

**Money question:** fan power is a direct line item in a system's power budget, and
acoustic limits are often a hard customer or regulatory requirement in telecom/embedded
hardware — not a nice-to-have. A control curve that's quieter and lower-power at the same
safety margin is a real product improvement, not just an academic exercise.

## The model

- `thermal_acoustic/workload.py` — a fixed synthetic heat-generation trace (Watts) with
  idle baseline + three burst windows, representing bursty compute load.
- `thermal_acoustic/simulate.py` — a simplified thermal model: a heat-balance ODE
  (Euler-integrated) where a fan removes heat proportional to its speed, plus fan-affinity
  power (`power ∝ speed³`) and superlinear acoustic scaling with speed. The fan-speed
  control curve is a piecewise-linear function of *current temperature* (closed-loop, not
  open-loop), interpolated between a handful of tunable breakpoints.
- `thermal_acoustic/objective.py` — mean power + mean noise, plus a large penalty
  (scaled by how far over) if the safety temperature limit is ever crossed.
- `thermal_acoustic/optimize.py` — a from-scratch (1+1)-evolution-strategy local search:
  perturb the current control curve with decaying-magnitude Gaussian noise, keep the
  perturbation only if it improves the score. A full optimization library would be
  overkill for a ~6-dimensional bounded problem like this one. It also includes a
  power/noise weight sweep and a safe efficiency/thermal-margin frontier extractor.

## Quickstart

```bash
pip install -r requirements.txt
python -m thermal_acoustic.cli --n-points 6 --iterations 500 --seed 0 --report reports/seed0.md
python -m thermal_acoustic.cli --n-points 6 --iterations 300 --seed 0 --pareto --report reports/pareto.md
```

## Honest results

At the default settings (6 control-curve breakpoints, 500 optimization iterations, seed 0):

| policy | mean power (W) | mean noise (dB) | max temp (°C) | safety violated |
| --- | ---: | ---: | ---: | :---: |
| always_on | 8.00 | 60.00 | 64.6 | no |
| linear_ramp | 2.55 | 43.57 | 73.1 | no |
| optimized | 1.08 | 35.34 | 85.0* | no |

\* exact value 84.99°C — the optimizer converges right up against the 85°C safety limit,
which is exactly the behavior you'd expect from a penalty-based optimizer: push the
constraint as far as it can go for free, then stop. That's not a coincidence or a rounding
artifact; it's the correct outcome for this objective, and it's worth having a smaller
safety margin than "right at the wall" if you were doing this for real (this simulation has
zero measurement noise or model uncertainty, unlike a real thermal system).

The optimized curve cuts mean power by **86%** and mean noise by **~25 dB perceived
range** (dB is logarithmic, so 60→35 is a large perceptual difference, not linear) relative
to `always_on`, while trading away 20°C of thermal margin to do it — a real, legible
tradeoff, not a free lunch.

One thing I did NOT get right on the first pass, worth being honest about: the initial
cooling-coefficient constant I picked made the safety limit **unreachable even at full fan
speed** (every policy showed as unsafe) — a modeling bug, not a control problem. Caught by
actually running the simulation before writing this table, not by inspection.

## Status / next steps

The project now supports a single optimized policy and a small efficiency/thermal-margin
Pareto sweep. The workload trace is still fixed and known in advance; a more realistic setup would
optimize against a *distribution* of workloads (or do online adaptation), and would model
measurement noise on the temperature sensor rather than assuming perfect state feedback.

## License

MIT — see [LICENSE](LICENSE).
