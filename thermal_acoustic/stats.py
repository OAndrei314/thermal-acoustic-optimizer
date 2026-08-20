"""Small-sample statistics helpers for the confidence-gated accept rule in
optimize.py: converting a nominal z-equivalent confidence level into a proper
Welch-Satterthwaite two-sample Student's-t critical value, instead of treating
a small-sample standard-error estimate as if it were exact (a normal/z test)."""
from __future__ import annotations

from scipy import stats


def welch_satterthwaite_df(se_a: float, n_a: int, se_b: float, n_b: int) -> float:
    """Effective degrees of freedom for the difference of two independent sample
    means with unequal variances (Welch's t-test), estimated from each side's
    standard error and sample size. Returns float('inf') when both sides have
    zero standard error (degenerate, e.g. deterministic scores) -- there is no
    sampling uncertainty left to correct for, so the caller should fall back to
    treating the estimate as exact."""
    numerator = (se_a**2 + se_b**2) ** 2
    denom = 0.0
    if n_a > 1:
        denom += se_a**4 / (n_a - 1)
    if n_b > 1:
        denom += se_b**4 / (n_b - 1)
    if denom == 0.0:
        return float("inf")
    return numerator / denom


def t_critical_from_z(z: float, df: float) -> float:
    """Convert a nominal one-sided normal critical value (z) into the Student's-t
    critical value at the same tail probability, for the given (possibly
    non-integer, Welch-Satterthwaite) degrees of freedom. As df -> infinity this
    converges to z itself; at small df it is strictly larger, which correctly
    makes the accept test more conservative when the standard error is itself
    only estimated from a handful of Monte-Carlo samples rather than known
    exactly -- the gap a fixed z critical value silently ignores."""
    if df == float("inf"):
        return z
    one_sided_tail = float(stats.norm.sf(z))
    return float(stats.t.ppf(1.0 - one_sided_tail, df))
