import numpy as np

from thermal_acoustic.workload import heat_trace, sample_heat_trace


def test_sample_heat_trace_is_reproducible_given_the_same_rng_state():
    trace_a = sample_heat_trace(np.random.default_rng(3))
    trace_b = sample_heat_trace(np.random.default_rng(3))
    assert np.array_equal(trace_a, trace_b)


def test_sample_heat_trace_varies_across_seeds():
    trace_a = sample_heat_trace(np.random.default_rng(0))
    trace_b = sample_heat_trace(np.random.default_rng(1))
    assert not np.array_equal(trace_a, trace_b)


def test_sample_heat_trace_has_the_same_shape_and_base_load_as_heat_trace():
    nominal = heat_trace()
    sampled = sample_heat_trace(np.random.default_rng(0))
    assert sampled.shape == nominal.shape
    assert sampled.min() >= 15.0 - 1e-9  # never drops below the idle base load


def test_sample_heat_trace_mean_is_close_to_the_nominal_fixed_trace():
    """sample_heat_trace() is meant to be a randomized draw from the same family
    heat_trace() sits in, not an unrelated distribution -- so its average over many
    draws should land near the fixed trace's own total heat, not systematically far
    from it."""
    nominal_total = heat_trace().sum()
    rng = np.random.default_rng(0)
    sampled_totals = [sample_heat_trace(rng).sum() for _ in range(200)]
    mean_total = float(np.mean(sampled_totals))
    assert abs(mean_total - nominal_total) / nominal_total < 0.15


def test_sample_heat_trace_respects_n_steps():
    sampled = sample_heat_trace(np.random.default_rng(0), n_steps=40)
    assert sampled.shape == (40,)


def test_sample_heat_trace_jitter_scale_zero_collapses_to_the_fixed_trace():
    """jitter_scale=0 should mean 'no workload uncertainty at all' -- no start jitter,
    duration and magnitude multipliers pinned to exactly 1.0 -- which is exactly what
    heat_trace() already is."""
    nominal = heat_trace()
    for seed in range(5):
        sampled = sample_heat_trace(np.random.default_rng(seed), jitter_scale=0.0)
        assert np.array_equal(sampled, nominal)


def test_sample_heat_trace_jitter_scale_one_matches_the_documented_defaults():
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    assert np.array_equal(sample_heat_trace(rng_a), sample_heat_trace(rng_b, jitter_scale=1.0))


def test_sample_heat_trace_larger_jitter_scale_widens_the_spread_of_outcomes():
    """A bigger jitter_scale should mean more workload uncertainty, not just different
    uncertainty -- so the variance of the total heat delivered across many draws should
    grow with jitter_scale."""
    def total_variance(scale: float) -> float:
        rng = np.random.default_rng(0)
        totals = [sample_heat_trace(rng, jitter_scale=scale).sum() for _ in range(300)]
        return float(np.var(totals))

    assert total_variance(0.5) < total_variance(1.0) < total_variance(2.0)
