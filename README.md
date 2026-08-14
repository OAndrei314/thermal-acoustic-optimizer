# thermal-acoustic-optimizer

*Maintained by: claude-actions-daily-routine · Status: Active*
A from-scratch simulation + local-search optimizer for a fan-speed control curve, trading
off power consumption and acoustic noise against a hard safety temperature limit — modeled
on real thermal/fan calibration work (power and acoustic optimization on telecom hardware),
fully synthetic and simplified, not a reproduction of any real chassis or dataset.

## Why this matters

**Research question:** starting from an always-on baseline and a naive linear ramp, how
much power and acoustic noise can a simple local-search optimizer save on a fixed
synthetic workload, without ever crossing the safety temperature limit?

**Practical impact:** fan power is a direct line item in a system's power budget, and
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
  power/noise weight sweep and a safe efficiency/thermal-margin frontier extractor, plus a
  noise-aware "robust" mode that scores each candidate as the mean over several noisy-sensor
  rollouts instead of one noiseless rollout.
- `thermal_acoustic/robustness.py` — Monte-Carlo evaluation of a *fixed* policy's safety
  under sensor read noise: how often does the true temperature actually cross the limit,
  not just what the noiseless objective predicts.

## Quickstart

```bash
pip install -r requirements.txt
python -m thermal_acoustic.cli --n-points 6 --iterations 500 --seed 0 --report reports/seed0.md
python -m thermal_acoustic.cli --n-points 6 --iterations 300 --seed 0 --pareto --report reports/pareto.md
python -m thermal_acoustic.cli --n-points 6 --iterations 500 --seed 0 \
    --sensor-noise-std 1.5 --noise-trials 300 --noise-trials-per-eval 20 \
    --report reports/seed0_robustness.md
python -m thermal_acoustic.cli --n-points 6 --iterations 500 --seed 0 \
    --sensor-noise-std 1.5 --noise-trials 300 --noise-trials-per-eval 20 \
    --compare-reevaluate-incumbent --report reports/seed0_reeval.md
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

### Sensor-noise robustness

The table above is noiseless: the controller reads the true temperature exactly. That's
unrealistic, and the README used to just flag it as a caveat ("this simulation has zero
measurement noise"). Now it's actually modeled: the controller only sees the true
temperature plus additive Gaussian read noise, while the real thermal state (and the
safety check) still uses the true, noiseless temperature — noise degrades the *decision*,
not the physics.

Re-evaluating the `optimized` policy above (still tuned assuming a perfect sensor) against
1.5°C of sensor noise, 300 Monte-Carlo trials, seed 0:

| policy | violation rate | mean max temp (°C) | worst max temp (°C) | mean power (W) |
| --- | ---: | ---: | ---: | ---: |
| optimized (noiseless-tuned) | 100.0% | 87.0 | 88.7 | 1.08 |
| robust_optimized (5 MC samples/eval) | 44.3% | 85.0 | 85.9 | 1.18 |
| robust_optimized (20 MC samples/eval) | 8.3% | 84.5 | 85.5 | 1.19 |
| robust_optimized (50 MC samples/eval) | 3.0% | 84.3 | 85.3 | — |

The `optimized` policy hugging the wall at 84.99°C means it violates the real safety limit
on *every single* noisy trial once you add realistic sensor noise — the exact fragility the
first-pass README predicted but never actually measured. Training the same optimizer
against noisy rollouts (`--sensor-noise-std`) fixes most of that at under a 10% power cost,
but doesn't fully eliminate it at the default 5-sample-per-candidate setting — the accept
criterion compares a candidate's noisy score against a *stale* score for the current best,
so a small sample size lets unlucky/lucky noise draws bias the walk. Increasing
`--noise-trials-per-eval` (more Monte Carlo samples per candidate evaluation) trades
optimization compute for a lower residual violation rate: 44%→8%→3% at 5→20→50 samples,
a real, honestly-measured, non-free tradeoff, not a fully solved problem.

### Fixing the stale-incumbent bias

The previous section's own theory was that the accept criterion compares a fresh candidate
score against a *stale* incumbent score, and that re-scoring the incumbent every iteration
(`reevaluate_incumbent=True` / CLI `--compare-reevaluate-incumbent`) should close some of
that gap. That was a hypothesis, not a measurement — so it's now implemented and tested
against the same `--sensor-noise-std 1.5 --noise-trials 300 --seed 0` setup used above:

| `--noise-trials-per-eval` | stale-incumbent violation rate | fresh-incumbent violation rate | power (stale → fresh) |
| ---: | ---: | ---: | :---: |
| 5 | 44.3% | **11.7%** | 1.16W → 1.18W |
| 20 | 8.3% | **0.7%** | 1.19W → 1.19W |
| 50 | 3.0% | **1.3%** | 1.21W → 1.22W |

Re-scoring the incumbent cuts the violation rate by roughly 3-12x at every sample budget
tested, for a power cost in the noise (≤0.02W). It isn't free, though: resampling the
incumbent doubles the simulation calls per iteration, so it's not simply "the same search,
fixed" — it's spending more compute per iteration in exchange for an unbiased accept
decision. To check it's not just "more compute wins," `stale @ 40 samples/eval` (the same
total simulation calls per iteration as `fresh @ 20`) was also measured: it lands at a 9.3%
violation rate, still ~13x worse than `fresh @ 20`'s 0.7% at matched compute — confirming
this is a real fix to a biased comparison, not just extra sampling.

Run it yourself with `--compare-reevaluate-incumbent` on the sensor-noise command above.

## Status / next steps

The project now supports a single optimized policy, a small efficiency/thermal-margin
Pareto sweep, noise-aware robust optimization against sensor read noise, and (as of this
change) a fresh-incumbent accept rule that fixes most of the residual safety-violation gap
that noise-aware optimization left open. What's left: the workload trace is still fixed and
known in advance; a more realistic setup would optimize against a *distribution* of
workloads (or do online adaptation) rather than one fixed trace. The fresh-incumbent fix
also still leaves a small non-zero violation rate at every sample budget tested (0.7-11.7%,
not 0%) — an explicit confidence-based acceptance rule (only accept a candidate when the
estimated score difference exceeds its standard error) would likely close more of that
residual than further raising the sample count, but hasn't been implemented or measured
here.

## License

MIT — see [LICENSE](LICENSE).
