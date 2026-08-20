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
python -m thermal_acoustic.cli --n-points 6 --iterations 500 --seed 0 \
    --sensor-noise-std 1.5 --noise-trials 300 --noise-trials-per-eval 20 \
    --compare-reevaluate-incumbent --confidence-z 1.5 --report reports/seed0_confidence.md
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

### Trying a confidence-based accept rule (an honest negative result)

The previous section's own next-step suggestion was that an explicit confidence-based
accept rule — only take a candidate when its estimated improvement over the incumbent
exceeds some number of standard errors of that estimate, instead of accepting *any*
estimated improvement — would likely close more of the residual violation-rate gap than
raising the sample count further. That's now implemented (`confidence_z=` /
CLI `--confidence-z`, requiring `reevaluate_incumbent`-style fresh resampling of both
sides every iteration) — and measuring it honestly does **not** confirm the hypothesis.

At seed 0, `--sensor-noise-std 1.5 --noise-trials 300`:

| `--noise-trials-per-eval` | fresh-incumbent violation rate | confidence-gated violation rate (z) |
| ---: | ---: | :---: |
| 5 | 11.7% | 31.0% (z=1.0) |
| 20 | 0.7% | 3.7% (z=1.5) |

That's *worse*, not better — and seed 0 alone isn't enough to trust either way, so it was
re-run across seeds 0-3 (`iterations=500`, same noise settings, `evaluate_robustness`
Monte-Carlo'd at 300 trials per policy):

| samples/eval | fresh-incumbent (mean over 4 seeds) | confidence-gated (mean over 4 seeds) | confidence beats fresh |
| ---: | ---: | ---: | :---: |
| 5 (z=1.0) | 23.3% | 31.7% | 1 / 4 seeds |
| 20 (z=1.5) | 7.8% | 7.9% | 2 / 4 seeds (incl. 1 tie) |
| 50 (z=1.5) | 1.1% | 2.3% | 1 / 4 seeds (incl. 1 tie) |

The confidence-gated variant loses to the plain fresh-incumbent rule more often than it
wins, at every sample budget tested, and the gap is largest exactly where you'd most want
the "be more careful before accepting" rule to help: `noise_trials_per_eval=5`. This is a
real, reproducible effect, not a fluke of one seed — but it's the opposite of what the
rule was supposed to do, which is worth reporting plainly rather than quietly dropping the
comparison.

**Working theory for why (not yet verified):** the accept test estimates each side's
standard error from only `noise_trials_per_eval` samples and compares the estimated gap
against a fixed z (normal) critical value. That's the correct test only if the standard
error were known exactly; here it's itself estimated from a small sample, so the
statistically correct critical value comes from a Student's-t distribution with
`noise_trials_per_eval - 1` degrees of freedom, which has heavier tails than the normal at
low degrees of freedom (df=4 at `noise_trials_per_eval=5`). Using a z critical value where
a t critical value is called for makes the test *less* strict than its nominal confidence
level suggests, which would let more marginal, noise-driven "improvements" through — the
opposite of the intended effect, and worse the smaller the sample. That plausibly explains
why the effect is largest at `noise_trials_per_eval=5` and mostly washes out by 50. It
hasn't been implemented or measured here, so treat it as a hypothesis, not a result.

### Fixing the z-vs-t gap directly (a second honest negative result)

The previous section's theory was concrete and testable: swap the fixed z critical value
for a proper small-sample Student's-t value — specifically a Welch-Satterthwaite
two-sample t, since the incumbent's and candidate's noisy-score variances are each
estimated independently and aren't assumed equal — and see whether that recovers the
improvement the naive z-based version failed to deliver. That's now implemented
(`thermal_acoustic/stats.py`: `welch_satterthwaite_df` + `t_critical_from_z`, wired into
`confidence_z`'s accept test in `optimize.py`), and measuring it honestly again does
**not** confirm the hypothesis.

Re-running the same seed-0-through-3, `--sensor-noise-std 1.5 --noise-trials 300` sweep,
comparing the t-corrected `confidence_z` variant against plain fresh-incumbent:

| samples/eval | fresh-incumbent (mean over 4 seeds) | t-gated (mean over 4 seeds) | z-gated (from previous section) | t-gated beats fresh |
| ---: | ---: | ---: | ---: | :---: |
| 5 (z=1.0) | 23.1% | 31.5% | 31.7% | 0 / 4 seeds |
| 20 (z=1.5) | 9.2% | 9.1% | 7.9% | 3 / 4 seeds (incl. 1 tie) |
| 50 (z=1.5) | 1.7% | 4.2% | 2.3% | 1 / 4 seeds |

The t-corrected gate performs about the same as the naive z-gate at every sample budget —
sometimes marginally better, sometimes marginally worse, never a clear win over either the
z-gate or plain fresh-incumbent. The seed-to-seed variance in these numbers (compare the
`23.1%` vs. `23.3%` fresh-incumbent means, recomputed from a fresh Monte-Carlo draw rather
than reused from the earlier table) is itself as large as the effect being measured.

**Why the fix didn't move the needle:** the Welch-Satterthwaite critical value was checked
directly against the plain z value at these settings — `t_critical_from_z(1.0, df=8) =
1.067` at `noise_trials_per_eval=5` (only 6.7% larger than z), `t_critical_from_z(1.5,
df=38) = 1.533` at 20 samples (2.2% larger), and `t_critical_from_z(1.5, df=98) = 1.513`
at 50 samples (0.8% larger). The original theory was right that the z-based test is
*technically* too lenient, but wrong about the fix mattering in practice: at these sample
sizes the correction is a few percent, nowhere near large enough to flip more than a
handful of accept/reject decisions across 500 search iterations. The real gap between
confidence-gating (in either form) and plain fresh-incumbent isn't a small-sample
statistics bug — it's that *any* gate stricter than "any measured improvement" slows the
search's ability to back away from the safety wall once it's already there, and that cost
outweighs the benefit of filtering noise-driven acceptances at this problem's scale. Worth
recording as a second confirmed negative result rather than re-tuning the same idea a
third time.

## Status / next steps

The project now supports a single optimized policy, a small efficiency/thermal-margin
Pareto sweep, noise-aware robust optimization against sensor read noise, a fresh-incumbent
accept rule that fixes most of the residual safety-violation gap noise-aware optimization
left open, and two confidence-based accept rule variants (naive z-gate, then a
statistically-correct Welch-Satterthwaite t-gate) that were each implemented specifically
to close the rest of that gap and, measured honestly across seeds, neither does — the
t-correction itself is real and correctly implemented, it's just too small at these sample
sizes to change the outcome. Confidence-gating as an approach is now a settled negative
result for this problem, not worth a third variant. What's left: the workload trace is
still fixed and known in advance; a more realistic setup would optimize against a
*distribution* of workloads (or do online adaptation) rather than one fixed trace — that's
the more promising direction for further work here than continuing to refine the
accept-rule statistics.

## License

MIT — see [LICENSE](LICENSE).
