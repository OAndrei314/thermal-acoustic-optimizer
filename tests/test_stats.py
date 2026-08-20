import math

from thermal_acoustic.stats import t_critical_from_z, welch_satterthwaite_df


def test_welch_satterthwaite_df_is_finite_and_positive_for_typical_inputs():
    df = welch_satterthwaite_df(se_a=0.5, n_a=5, se_b=0.3, n_b=5)
    assert df > 0
    assert math.isfinite(df)


def test_welch_satterthwaite_df_equal_variance_equal_n_matches_pooled_df():
    """With equal standard errors and equal sample sizes on both sides, the
    Welch-Satterthwaite formula reduces exactly to the pooled two-sample df,
    2*(n-1) -- a known special case worth pinning down directly."""
    se = 0.4
    n = 10
    df = welch_satterthwaite_df(se, n, se, n)
    assert math.isclose(df, 2 * (n - 1))


def test_welch_satterthwaite_df_is_infinite_when_both_sides_are_deterministic():
    assert welch_satterthwaite_df(0.0, 5, 0.0, 5) == float("inf")


def test_welch_satterthwaite_df_ignores_n_when_that_side_has_zero_variance():
    """A side with se=0 contributes nothing to the denominator regardless of its
    n (n=1 would otherwise divide by zero) -- only the noisy side's sampling
    uncertainty should determine df."""
    df = welch_satterthwaite_df(se_a=0.5, n_a=5, se_b=0.0, n_b=1)
    assert math.isfinite(df)
    assert df > 0


def test_t_critical_from_z_zero_is_zero_regardless_of_df():
    """The t distribution is symmetric about 0 at every df, so the median (and
    hence the 50th-percentile critical value for z=0) is exactly 0 in theory --
    allow a tiny numerical tolerance since scipy's t.ppf inverts a CDF
    iteratively and isn't bit-exact across platforms/versions, especially at
    df=1 (the Cauchy distribution, notoriously ill-conditioned near its
    median)."""
    for df in (1, 4, 19, 49, 1000):
        assert math.isclose(t_critical_from_z(0.0, df), 0.0, abs_tol=1e-9)


def test_t_critical_from_z_matches_z_at_infinite_df():
    for z in (0.5, 1.0, 1.5, 2.0):
        assert math.isclose(t_critical_from_z(z, float("inf")), z)


def test_t_critical_from_z_is_more_conservative_at_low_df():
    """The whole point of the fix: at small degrees of freedom, the t critical
    value must be strictly larger than the naive z value, since the standard
    error itself is only estimated from a handful of samples."""
    z = 1.5
    low_df_crit = t_critical_from_z(z, df=4)
    high_df_crit = t_critical_from_z(z, df=200)
    assert low_df_crit > z
    assert low_df_crit > high_df_crit
    assert math.isclose(high_df_crit, z, abs_tol=0.05)


def test_t_critical_from_z_decreases_monotonically_toward_z_as_df_grows():
    z = 1.0
    crits = [t_critical_from_z(z, df) for df in (2, 5, 10, 30, 100, 1000)]
    assert all(a >= b for a, b in zip(crits, crits[1:]))
    assert all(c >= z for c in crits)


def test_t_critical_from_z_extreme_threshold_is_effectively_unreachable():
    crit = t_critical_from_z(1000.0, df=4)
    assert crit == float("inf") or crit > 1e6
