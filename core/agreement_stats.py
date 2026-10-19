"""Agreement statistics for criterion validation (pure Python, no SciPy).

The functions compare an automated measure with a reference (typically blinded
manual scoring) and return small dataclasses that carry the value together with
the inputs needed to report it (sample size, confidence interval, p-value...).

Only the standard library and NumPy are used: ``scipy.stats`` is not shipped
in the packaged application. The t and F distributions are built from a
regularized incomplete beta function (continued fraction, as in Numerical
Recipes) and quantiles are found by bisection, so p-values and confidence
intervals are exact to numerical precision rather than normal approximations.

Conventions
-----------
* "a" is the automated measure and "b" the reference (manual) measure wherever
  the order matters: differences are ``a - b``, so a positive Bland-Altman bias
  means the automated score over-estimates.
* Undefined statistics (for example kappa when both raters use one category)
  are returned as ``nan`` rather than raising, so a batch report can continue.
  Inputs that are malformed (unequal lengths, too few raters) raise
  ``ValueError``.
* Confidence intervals are two-sided and default to 95 %.
"""
from dataclasses import dataclass
import math
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np


NAN = float("nan")


# --------------------------------------------------------------------------
# Distribution functions
# --------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 2000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a, b)."""
    if a <= 0 or b <= 0:
        raise ValueError("betainc requires a > 0 and b > 0")
    if x < 0.0 or x > 1.0:
        raise ValueError("betainc requires 0 <= x <= 1")
    if x == 0.0 or x == 1.0:
        return x
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    """Cumulative distribution function of Student's t."""
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    if math.isinf(t):
        return 1.0 if t > 0 else 0.0
    x = df / (df + t * t)
    tail = 0.5 * betainc(df / 2.0, 0.5, x)
    return 1.0 - tail if t > 0 else tail


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p-value for a t statistic."""
    if math.isnan(t):
        return NAN
    if math.isinf(t):
        return 0.0
    return betainc(df / 2.0, 0.5, df / (df + t * t))


def f_cdf(f: float, d1: float, d2: float) -> float:
    """Cumulative distribution function of the F distribution."""
    if d1 <= 0 or d2 <= 0:
        raise ValueError("degrees of freedom must be positive")
    if f <= 0:
        return 0.0
    if math.isinf(f):
        return 1.0
    return betainc(d1 / 2.0, d2 / 2.0, d1 * f / (d1 * f + d2))


def _bisect_quantile(cdf, p: float, lower: float, upper_start: float) -> float:
    """Invert a monotone CDF on ``[lower, inf)`` by bracketing and bisection."""
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    low, high = lower, upper_start
    while cdf(high) < p:
        low, high = high, high * 2.0
        if high > 1e300:
            return math.inf
    for _ in range(200):
        mid = 0.5 * (low + high)
        if cdf(mid) < p:
            low = mid
        else:
            high = mid
        if high - low <= 1e-13 * max(1.0, abs(high)):
            break
    return 0.5 * (low + high)


def t_ppf(p: float, df: float) -> float:
    """Quantile (inverse CDF) of Student's t."""
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df)
    return _bisect_quantile(lambda value: t_cdf(value, df), p, 0.0, 8.0)


def f_ppf(p: float, d1: float, d2: float) -> float:
    """Quantile (inverse CDF) of the F distribution."""
    return _bisect_quantile(lambda value: f_cdf(value, d1, d2), p, 0.0, 8.0)


def norm_cdf(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Standard normal quantile (bisection on :func:`norm_cdf`)."""
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    low, high = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if norm_cdf(mid) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _check_confidence(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    return confidence


def _paired_arrays(a, b) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(
            f"Paired inputs must have the same length ({x.size} != {y.size})"
        )
    return x, y


# --------------------------------------------------------------------------
# Cohen's kappa
# --------------------------------------------------------------------------

@dataclass
class KappaResult:
    """Unweighted Cohen's kappa with its inputs.

    ``kappa`` is ``nan`` when it is undefined (no items, or chance agreement
    equal to 1 because both raters used a single category).
    ``confusion`` has rows for ``labels_a`` and columns for ``labels_b``, in
    the order of ``categories``.
    """
    kappa: float
    n: int
    observed_agreement: float
    expected_agreement: float
    se: float
    ci_low: float
    ci_high: float
    confidence: float
    categories: List[Hashable]
    confusion: np.ndarray


def cohens_kappa(
    labels_a: Sequence[Hashable],
    labels_b: Sequence[Hashable],
    categories: Optional[Sequence[Hashable]] = None,
    confidence: float = 0.95,
) -> KappaResult:
    """Unweighted Cohen's kappa for two raters over the same items.

    The standard error is the large-sample estimate
    ``sqrt(po * (1 - po) / (n * (1 - pe) ** 2))`` and the interval is
    ``kappa +/- z * se`` clipped to [-1, 1].
    """
    _check_confidence(confidence)
    a, b = list(labels_a), list(labels_b)
    if len(a) != len(b):
        raise ValueError(
            f"Label sequences must have the same length ({len(a)} != {len(b)})"
        )
    if categories is None:
        seen = list(dict.fromkeys(a + b))
        try:
            categories = sorted(seen)
        except TypeError:
            categories = seen
    categories = list(categories)
    index = {category: i for i, category in enumerate(categories)}
    unknown = {label for label in a + b if label not in index}
    if unknown:
        raise ValueError(f"Labels not in categories: {sorted(map(str, unknown))}")

    size = len(categories)
    confusion = np.zeros((size, size), dtype=int)
    for left, right in zip(a, b):
        confusion[index[left], index[right]] += 1
    n = len(a)
    if n == 0:
        return KappaResult(NAN, 0, NAN, NAN, NAN, NAN, NAN, confidence,
                           categories, confusion)

    po = float(np.trace(confusion)) / n
    row = confusion.sum(axis=1) / n
    col = confusion.sum(axis=0) / n
    pe = float(np.dot(row, col))
    if pe >= 1.0:
        return KappaResult(NAN, n, po, pe, NAN, NAN, NAN, confidence,
                           categories, confusion)
    kappa = (po - pe) / (1.0 - pe)
    se = math.sqrt(po * (1.0 - po) / (n * (1.0 - pe) ** 2))
    z = norm_ppf(0.5 + confidence / 2.0)
    return KappaResult(
        kappa, n, po, pe, se,
        max(-1.0, kappa - z * se), min(1.0, kappa + z * se),
        confidence, categories, confusion,
    )


# --------------------------------------------------------------------------
# Intraclass correlation
# --------------------------------------------------------------------------

@dataclass
class ICCResult:
    """Intraclass correlation (single rater) with ANOVA mean squares."""
    icc: float
    model: str
    n_subjects: int
    n_raters: int
    ci_low: float
    ci_high: float
    confidence: float
    f_value: float
    p_value: float
    ms_rows: float
    ms_columns: float
    ms_error: float


def icc(
    ratings,
    model: str = "2,1",
    confidence: float = 0.95,
) -> ICCResult:
    """Single-rater intraclass correlation from a subjects x raters matrix.

    ``model="2,1"`` is the two-way random-effects, absolute-agreement ICC;
    ``model="3,1"`` is the two-way mixed-effects, consistency ICC (Shrout &
    Fleiss, 1979). Confidence intervals follow Shrout & Fleiss for (3,1) and
    McGraw & Wong (1996) for (2,1), using exact F quantiles.

    When every rater agrees perfectly (error mean square 0) the ICC is 1 and
    the interval is undefined (nan).
    """
    if model not in ("2,1", "3,1"):
        raise ValueError("model must be '2,1' or '3,1'")
    _check_confidence(confidence)
    data = np.asarray(ratings, dtype=float)
    if data.ndim != 2:
        raise ValueError("ratings must be a 2-D subjects x raters array")
    n, k = data.shape
    if n < 2 or k < 2:
        raise ValueError("ICC needs at least 2 subjects and 2 raters")
    if not np.all(np.isfinite(data)):
        raise ValueError("ratings must be finite (drop incomplete subjects)")

    grand = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ss_rows = k * float(np.sum((row_means - grand) ** 2))
    ss_cols = n * float(np.sum((col_means - grand) ** 2))
    ss_total = float(np.sum((data - grand) ** 2))
    ss_error = max(0.0, ss_total - ss_rows - ss_cols)
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_error / ((n - 1) * (k - 1))

    # Scale-aware tolerance for "zero" error variance.
    scale = max(msr, msc, abs(grand), 1.0)
    if mse <= 1e-12 * scale:
        mse = 0.0
    df1, df2 = n - 1, (n - 1) * (k - 1)

    if model == "3,1":
        denom = msr + (k - 1) * mse
    else:
        denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom <= 0.0:
        value = NAN
    else:
        value = (msr - mse) / denom

    if mse == 0.0:
        f_value = math.inf if msr > 0 else NAN
        p_value = 0.0 if msr > 0 else NAN
        return ICCResult(value, model, n, k, NAN, NAN, confidence, f_value,
                         p_value, msr, msc, mse)

    f_value = msr / mse
    p_value = 1.0 - f_cdf(f_value, df1, df2)
    alpha = 1.0 - confidence
    q_hi = 1.0 - alpha / 2.0
    if model == "3,1":
        fl = f_value / f_ppf(q_hi, df1, df2)
        fu = f_value * f_ppf(q_hi, df2, df1)
        low = (fl - 1.0) / (fl + k - 1.0)
        high = (fu - 1.0) / (fu + k - 1.0)
    else:
        rho = value
        a = k * rho / (n * (1.0 - rho)) if rho < 1.0 else math.inf
        b = 1.0 + k * rho * (n - 1.0) / (n * (1.0 - rho)) if rho < 1.0 else math.inf
        if not (math.isfinite(a) and math.isfinite(b)):
            low = high = NAN
        else:
            v_num = (a * msc + b * mse) ** 2
            v_den = (a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1))
            v = v_num / v_den
            f_star = f_ppf(q_hi, df1, v)
            f_star2 = f_ppf(q_hi, v, df1)
            common = k * msc + (k * n - k - n) * mse
            low = n * (msr - f_star * mse) / (f_star * common + n * msr)
            high = n * (f_star2 * msr - mse) / (common + n * f_star2 * msr)
    return ICCResult(
        value, model, n, k, max(-1.0, low), min(1.0, high), confidence,
        f_value, p_value, msr, msc, mse,
    )


# --------------------------------------------------------------------------
# Bland-Altman
# --------------------------------------------------------------------------

@dataclass
class BlandAltmanResult:
    """Bland-Altman agreement statistics for differences ``a - b``.

    ``slope`` is the regression of the difference on the pair mean; a slope
    whose interval excludes 0 (``slope_p_value`` small) signals proportional
    bias. ``loa_*_ci`` are approximate 95 % intervals (Bland & Altman, 1986).
    """
    n: int
    bias: float
    sd_diff: float
    bias_ci: Tuple[float, float]
    loa_lower: float
    loa_upper: float
    loa_lower_ci: Tuple[float, float]
    loa_upper_ci: Tuple[float, float]
    slope: float
    intercept: float
    slope_ci: Tuple[float, float]
    slope_p_value: float
    confidence: float
    differences: np.ndarray
    means: np.ndarray


def bland_altman(a, b, confidence: float = 0.95) -> BlandAltmanResult:
    """Bland-Altman bias, limits of agreement and proportional-bias slope."""
    _check_confidence(confidence)
    x, y = _paired_arrays(a, b)
    n = int(x.size)
    nan_pair = (NAN, NAN)
    if n < 2:
        return BlandAltmanResult(
            n, float(np.mean(x - y)) if n else NAN, NAN, nan_pair, NAN, NAN,
            nan_pair, nan_pair, NAN, NAN, nan_pair, NAN, confidence,
            x - y, (x + y) / 2.0,
        )
    diff = x - y
    mean = (x + y) / 2.0
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    z = norm_ppf(0.5 + confidence / 2.0)
    t_crit = t_ppf(0.5 + confidence / 2.0, n - 1)
    se_bias = sd / math.sqrt(n)
    se_loa = math.sqrt(sd * sd * (1.0 / n + z * z / (2.0 * (n - 1))))
    lower, upper = bias - z * sd, bias + z * sd

    slope = intercept = slope_p = NAN
    slope_ci = nan_pair
    sxx = float(np.sum((mean - mean.mean()) ** 2))
    if n >= 3 and sxx > 0.0:
        sxy = float(np.sum((mean - mean.mean()) * (diff - bias)))
        slope = sxy / sxx
        intercept = bias - slope * float(mean.mean())
        residual = diff - (intercept + slope * mean)
        sse = float(np.sum(residual ** 2))
        se_slope = math.sqrt(sse / (n - 2) / sxx)
        t_slope = (
            slope / se_slope if se_slope > 0
            else (math.inf if slope else 0.0)
        )
        slope_p = t_two_sided_p(t_slope, n - 2)
        t_slope_crit = t_ppf(0.5 + confidence / 2.0, n - 2)
        slope_ci = (slope - t_slope_crit * se_slope,
                    slope + t_slope_crit * se_slope)
    return BlandAltmanResult(
        n=n, bias=bias, sd_diff=sd,
        bias_ci=(bias - t_crit * se_bias, bias + t_crit * se_bias),
        loa_lower=lower, loa_upper=upper,
        loa_lower_ci=(lower - t_crit * se_loa, lower + t_crit * se_loa),
        loa_upper_ci=(upper - t_crit * se_loa, upper + t_crit * se_loa),
        slope=slope, intercept=intercept, slope_ci=slope_ci,
        slope_p_value=slope_p, confidence=confidence,
        differences=diff, means=mean,
    )


# --------------------------------------------------------------------------
# Correlation
# --------------------------------------------------------------------------

@dataclass
class CorrelationResult:
    """Correlation coefficient with a Fisher-z interval and two-sided p."""
    method: str
    r: float
    n: int
    ci_low: float
    ci_high: float
    p_value: float
    confidence: float


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks, ties receiving the mean of the ranks they span."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start
        while end + 1 < values.size and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[order[start:end + 1]] = (start + end) / 2.0 + 1.0
        start = end + 1
    return ranks


def _correlation(method, x, y, confidence, se_factor) -> CorrelationResult:
    n = int(x.size)
    if n < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return CorrelationResult(method, NAN, n, NAN, NAN, NAN, confidence)
    r = float(np.corrcoef(x, y)[0, 1])
    r = max(-1.0, min(1.0, r))
    if abs(r) >= 1.0:
        return CorrelationResult(method, r, n, r, r, 0.0, confidence)
    t = r * math.sqrt((n - 2) / (1.0 - r * r))
    p_value = t_two_sided_p(t, n - 2)
    if n < 4:
        low = high = NAN
    else:
        z = norm_ppf(0.5 + confidence / 2.0)
        se = math.sqrt(se_factor(r)) / math.sqrt(n - 3)
        centre = math.atanh(r)
        low, high = math.tanh(centre - z * se), math.tanh(centre + z * se)
    return CorrelationResult(method, r, n, low, high, p_value, confidence)


def pearson(a, b, confidence: float = 0.95) -> CorrelationResult:
    """Pearson r with Fisher-z interval and t-based two-sided p-value."""
    _check_confidence(confidence)
    x, y = _paired_arrays(a, b)
    return _correlation("pearson", x, y, confidence, lambda r: 1.0)


def spearman(a, b, confidence: float = 0.95) -> CorrelationResult:
    """Spearman rho (average ranks for ties).

    The interval uses the Bonett-Wright variance ``(1 + rho^2 / 2) / (n - 3)``
    on the Fisher-z scale; the p-value is the t approximation.
    """
    _check_confidence(confidence)
    x, y = _paired_arrays(a, b)
    return _correlation(
        "spearman", _average_ranks(x), _average_ranks(y), confidence,
        lambda r: 1.0 + r * r / 2.0,
    )


# --------------------------------------------------------------------------
# Error metrics
# --------------------------------------------------------------------------

def rmse(a, b) -> float:
    """Root mean squared error between paired values (nan when empty)."""
    x, y = _paired_arrays(a, b)
    if x.size == 0:
        return NAN
    return float(math.sqrt(np.mean((x - y) ** 2)))


def mae(a, b) -> float:
    """Mean absolute error between paired values (nan when empty)."""
    x, y = _paired_arrays(a, b)
    if x.size == 0:
        return NAN
    return float(np.mean(np.abs(x - y)))


def mean_absolute_percentage_difference(a, b, denominator: str = "mean") -> float:
    """Mean absolute percentage difference between paired values.

    ``denominator="mean"`` (default) divides ``|a - b|`` by the mean of
    ``|a|`` and ``|b|``, which is symmetric and bounded by 200 %.
    ``denominator="b"`` divides by ``|b|`` (error relative to the reference).
    Pairs whose denominator is zero are skipped; nan when none remain.
    """
    if denominator not in ("mean", "b"):
        raise ValueError("denominator must be 'mean' or 'b'")
    x, y = _paired_arrays(a, b)
    if denominator == "mean":
        scale = (np.abs(x) + np.abs(y)) / 2.0
    else:
        scale = np.abs(y)
    keep = scale > 0
    if not np.any(keep):
        return NAN
    return float(np.mean(np.abs(x - y)[keep] / scale[keep]) * 100.0)


def percent_agreement(labels_a: Sequence[Hashable], labels_b: Sequence[Hashable]) -> float:
    """Percentage of items with identical labels (nan when empty)."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Label sequences must have the same length")
    if not len(labels_a):
        return NAN
    same = sum(left == right for left, right in zip(labels_a, labels_b))
    return same / len(labels_a) * 100.0


# --------------------------------------------------------------------------
# Batch-level summary
# --------------------------------------------------------------------------

@dataclass
class AgreementSummary:
    """Agreement of one per-video metric between automated and manual scoring.

    ``n`` counts the pairs used; ``n_excluded`` those dropped because either
    value was missing (None or nan). ``icc`` is ``None`` when fewer than two
    pairs remain. Differences follow the ``auto - manual`` convention.
    """
    n: int
    n_excluded: int
    mean_auto: float
    mean_manual: float
    icc: Optional[ICCResult]
    bland_altman: BlandAltmanResult
    pearson: CorrelationResult
    spearman: CorrelationResult
    rmse: float
    mae: float
    mapd: float


def clean_pairs(pairs) -> Tuple[np.ndarray, np.ndarray, int]:
    """Split ``(auto, manual)`` pairs into arrays, dropping missing values.

    Returns ``(auto, manual, n_excluded)``.
    """
    auto, manual, excluded = [], [], 0
    for pair in pairs:
        left, right = pair
        try:
            left_value = float(left)
            right_value = float(right)
        except (TypeError, ValueError):
            excluded += 1
            continue
        if not (math.isfinite(left_value) and math.isfinite(right_value)):
            excluded += 1
            continue
        auto.append(left_value)
        manual.append(right_value)
    return np.asarray(auto), np.asarray(manual), excluded


def summarize_agreement(
    per_video_pairs,
    confidence: float = 0.95,
) -> AgreementSummary:
    """ICC(2,1), Bland-Altman, Pearson and Spearman for per-video pairs.

    ``per_video_pairs`` is a sequence of ``(auto, manual)`` values, one per
    video, for a single metric (for example the Y-maze alternation percentage).
    Pairs with a missing value are dropped and counted in ``n_excluded``.
    """
    auto, manual, excluded = clean_pairs(per_video_pairs)
    n = int(auto.size)
    icc_result = None
    if n >= 2:
        icc_result = icc(np.column_stack([auto, manual]), "2,1", confidence)
    return AgreementSummary(
        n=n,
        n_excluded=excluded,
        mean_auto=float(auto.mean()) if n else NAN,
        mean_manual=float(manual.mean()) if n else NAN,
        icc=icc_result,
        bland_altman=bland_altman(auto, manual, confidence),
        pearson=pearson(auto, manual, confidence),
        spearman=spearman(auto, manual, confidence),
        rmse=rmse(auto, manual),
        mae=mae(auto, manual),
        mapd=mean_absolute_percentage_difference(auto, manual),
    )


def summarize_metrics(per_metric_pairs, confidence: float = 0.95) -> Dict[str, AgreementSummary]:
    """:func:`summarize_agreement` for each ``{metric: pairs}`` entry."""
    return {
        name: summarize_agreement(pairs, confidence)
        for name, pairs in per_metric_pairs.items()
    }


__all__ = [
    "AgreementSummary", "BlandAltmanResult", "CorrelationResult", "ICCResult", "KappaResult",
    "bland_altman", "betainc", "clean_pairs", "cohens_kappa", "f_cdf", "f_ppf", "icc",
    "mae", "mean_absolute_percentage_difference", "norm_cdf", "norm_ppf",
    "pearson", "percent_agreement", "rmse", "spearman", "summarize_agreement", "summarize_metrics", "t_cdf", "t_ppf",
    "t_two_sided_p",
]
