import math
import unittest

import numpy as np

from core import agreement_stats as stats


# Shrout & Fleiss (1979), Table 2: 6 subjects rated by 4 judges.
SHROUT_FLEISS = [
    [9, 2, 5, 8],
    [6, 1, 3, 2],
    [8, 4, 6, 8],
    [7, 1, 2, 6],
    [10, 5, 6, 9],
    [6, 2, 4, 7],
]

# Bland & Altman (1986): peak expiratory flow rate (l/min), first reading
# with the Wright peak flow meter and the mini Wright meter, 17 subjects.
PEFR_WRIGHT = [494, 395, 516, 434, 476, 557, 413, 442, 650, 433,
               417, 656, 267, 478, 178, 423, 427]
PEFR_MINI = [512, 430, 520, 428, 500, 600, 364, 380, 658, 445,
             432, 626, 260, 477, 259, 350, 451]


class DistributionTests(unittest.TestCase):
    def test_incomplete_beta_known_values(self):
        self.assertAlmostEqual(stats.betainc(1, 1, 0.3), 0.3, places=12)
        self.assertAlmostEqual(stats.betainc(2, 1, 0.5), 0.25, places=12)
        self.assertAlmostEqual(stats.betainc(1, 2, 0.5), 0.75, places=12)
        self.assertAlmostEqual(stats.betainc(0.5, 0.5, 0.5), 0.5, places=12)
        # I_x(a, b) = 1 - I_{1-x}(b, a)
        self.assertAlmostEqual(
            stats.betainc(3.5, 7.25, 0.2),
            1.0 - stats.betainc(7.25, 3.5, 0.8), places=12)
        # Arcsine law: I_x(1/2, 1/2) = (2 / pi) * asin(sqrt(x))
        self.assertAlmostEqual(
            stats.betainc(0.5, 0.5, 0.2),
            2 / math.pi * math.asin(math.sqrt(0.2)), places=12)
        self.assertEqual(stats.betainc(2, 3, 0.0), 0.0)
        self.assertEqual(stats.betainc(2, 3, 1.0), 1.0)
        with self.assertRaises(ValueError):
            stats.betainc(0, 1, 0.5)
        with self.assertRaises(ValueError):
            stats.betainc(1, 1, 1.5)

    def test_t_cdf_and_quantiles_match_tables(self):
        self.assertAlmostEqual(stats.t_cdf(0.0, 7), 0.5, places=12)
        # df = 1 is the Cauchy distribution.
        self.assertAlmostEqual(
            stats.t_cdf(1.0, 1), 0.5 + math.atan(1.0) / math.pi, places=12)
        self.assertAlmostEqual(stats.t_ppf(0.975, 10), 2.228, places=3)
        self.assertAlmostEqual(stats.t_ppf(0.975, 1), 12.706, places=3)
        self.assertAlmostEqual(stats.t_ppf(0.975, 30), 2.042, places=3)
        self.assertAlmostEqual(stats.t_ppf(0.995, 5), 4.032, places=3)
        self.assertAlmostEqual(stats.t_ppf(0.025, 10), -2.228, places=3)
        self.assertAlmostEqual(stats.t_ppf(0.975, 1000), 1.962, places=3)
        self.assertAlmostEqual(
            stats.t_cdf(stats.t_ppf(0.9, 6), 6), 0.9, places=10)

    def test_two_sided_p_value(self):
        self.assertAlmostEqual(
            stats.t_two_sided_p(2.228138852, 10), 0.05, places=6)
        self.assertEqual(stats.t_two_sided_p(0.0, 5), 1.0)
        self.assertEqual(stats.t_two_sided_p(math.inf, 5), 0.0)

    def test_f_quantiles_match_tables(self):
        self.assertAlmostEqual(stats.f_ppf(0.975, 5, 18), 3.38, places=2)
        self.assertAlmostEqual(stats.f_ppf(0.95, 3, 10), 3.708, places=3)
        self.assertAlmostEqual(stats.f_ppf(0.95, 2, 2), 19.0, places=6)
        self.assertAlmostEqual(stats.f_ppf(0.975, 1, 1), 647.79, places=2)
        self.assertAlmostEqual(stats.f_ppf(0.975, 10, 10), 3.717, places=3)
        self.assertAlmostEqual(
            stats.f_cdf(stats.f_ppf(0.9, 4, 9), 4, 9), 0.9, places=10)
        # F(1, d) is t squared.
        self.assertAlmostEqual(
            stats.f_ppf(0.95, 1, 12), stats.t_ppf(0.975, 12) ** 2, places=6)

    def test_normal_quantile(self):
        self.assertAlmostEqual(stats.norm_ppf(0.975), 1.959964, places=6)
        self.assertAlmostEqual(stats.norm_ppf(0.5), 0.0, places=9)
        self.assertAlmostEqual(stats.norm_cdf(1.96), 0.975, places=4)


class KappaTests(unittest.TestCase):
    def test_textbook_two_by_two_example(self):
        # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15
        # (observed agreement 0.7, chance agreement 0.5).
        a = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
        b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
        result = stats.cohens_kappa(a, b)
        self.assertEqual(result.n, 50)
        self.assertAlmostEqual(result.observed_agreement, 0.7)
        self.assertAlmostEqual(result.expected_agreement, 0.5)
        self.assertAlmostEqual(result.kappa, 0.4)
        self.assertAlmostEqual(result.se, math.sqrt(0.7 * 0.3 / (50 * 0.25)))
        self.assertAlmostEqual(
            result.ci_low, 0.4 - 1.959964 * result.se, places=5)
        self.assertAlmostEqual(
            result.ci_high, 0.4 + 1.959964 * result.se, places=5)
        self.assertEqual(result.categories, ["n", "y"])
        self.assertEqual(result.confusion.tolist(), [[15, 10], [5, 20]])

    def test_perfect_agreement_across_categories(self):
        labels = ["a", "b", "c", "a", "b"]
        result = stats.cohens_kappa(labels, labels)
        self.assertAlmostEqual(result.kappa, 1.0)
        self.assertEqual(result.se, 0.0)
        self.assertEqual((result.ci_low, result.ci_high), (1.0, 1.0))

    def test_single_category_is_undefined(self):
        result = stats.cohens_kappa(["x"] * 4, ["x"] * 4)
        self.assertTrue(math.isnan(result.kappa))
        self.assertEqual(result.observed_agreement, 1.0)
        self.assertEqual(result.expected_agreement, 1.0)

    def test_empty_input_is_undefined(self):
        result = stats.cohens_kappa([], [])
        self.assertEqual(result.n, 0)
        self.assertTrue(math.isnan(result.kappa))

    def test_explicit_categories_shape_the_confusion_matrix(self):
        result = stats.cohens_kappa(
            ["a", "b"], ["a", "a"], categories=["a", "b", "c"])
        self.assertEqual(result.confusion.shape, (3, 3))
        self.assertEqual(result.confusion[1, 0], 1)

    def test_validation_errors(self):
        with self.assertRaises(ValueError):
            stats.cohens_kappa(["a"], ["a", "b"])
        with self.assertRaises(ValueError):
            stats.cohens_kappa(["a", "z"], ["a", "a"], categories=["a"])

    def test_chance_level_agreement_is_zero(self):
        a = ["y", "y", "n", "n"]
        b = ["y", "n", "y", "n"]
        self.assertAlmostEqual(stats.cohens_kappa(a, b).kappa, 0.0)

    def test_systematic_disagreement_is_negative(self):
        result = stats.cohens_kappa(["y", "n", "y", "n"], ["n", "y", "n", "y"])
        self.assertAlmostEqual(result.kappa, -1.0)


class ICCTests(unittest.TestCase):
    def test_shrout_fleiss_values(self):
        two_one = stats.icc(SHROUT_FLEISS, "2,1")
        three_one = stats.icc(SHROUT_FLEISS, "3,1")
        self.assertAlmostEqual(two_one.icc, 0.29, places=2)
        self.assertAlmostEqual(three_one.icc, 0.71, places=2)
        # Mean squares in the paper's ANOVA table.
        self.assertAlmostEqual(two_one.ms_rows, 11.24, places=2)
        self.assertAlmostEqual(two_one.ms_columns, 32.49, places=2)
        self.assertAlmostEqual(two_one.ms_error, 1.02, places=2)
        self.assertEqual((two_one.n_subjects, two_one.n_raters), (6, 4))

    def test_shrout_fleiss_confidence_intervals(self):
        two_one = stats.icc(SHROUT_FLEISS, "2,1")
        self.assertAlmostEqual(two_one.ci_low, 0.02, delta=0.01)
        self.assertAlmostEqual(two_one.ci_high, 0.76, delta=0.01)
        three_one = stats.icc(SHROUT_FLEISS, "3,1")
        self.assertAlmostEqual(three_one.ci_low, 0.34, delta=0.01)
        self.assertAlmostEqual(three_one.ci_high, 0.95, delta=0.01)
        self.assertLess(two_one.p_value, 0.001)

    def test_perfect_agreement(self):
        data = [[1, 1], [2, 2], [5, 5], [9, 9]]
        for model in ("2,1", "3,1"):
            result = stats.icc(data, model)
            self.assertAlmostEqual(result.icc, 1.0)
            self.assertTrue(math.isnan(result.ci_low))

    def test_systematic_offset_separates_models(self):
        # Rater 2 reads exactly 10 higher: consistency is perfect,
        # absolute agreement is not.
        base = [3.0, 7.0, 1.0, 9.0, 5.0, 4.0]
        data = [[v, v + 10.0] for v in base]
        self.assertAlmostEqual(stats.icc(data, "3,1").icc, 1.0)
        self.assertLess(stats.icc(data, "2,1").icc, 0.6)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            stats.icc([[1, 2]], "2,1")
        with self.assertRaises(ValueError):
            stats.icc([[1], [2]], "2,1")
        with self.assertRaises(ValueError):
            stats.icc([1, 2, 3], "2,1")
        with self.assertRaises(ValueError):
            stats.icc(SHROUT_FLEISS, "1,1")
        with self.assertRaises(ValueError):
            stats.icc([[1, float("nan")], [2, 3]], "2,1")

    def test_interval_contains_estimate(self):
        rng = np.random.default_rng(3)
        truth = rng.normal(10, 3, 30)
        data = np.column_stack([truth + rng.normal(0, 1, 30),
                                truth + rng.normal(0, 1, 30)])
        for model in ("2,1", "3,1"):
            result = stats.icc(data, model)
            self.assertLess(result.ci_low, result.icc)
            self.assertGreater(result.ci_high, result.icc)
            self.assertGreater(result.icc, 0.7)


class BlandAltmanTests(unittest.TestCase):
    def test_pefr_published_values(self):
        result = stats.bland_altman(PEFR_WRIGHT, PEFR_MINI)
        self.assertEqual(result.n, 17)
        self.assertAlmostEqual(result.bias, -2.1, places=1)
        self.assertAlmostEqual(result.sd_diff, 38.8, places=1)
        self.assertAlmostEqual(result.loa_lower, -78.1, places=1)
        self.assertAlmostEqual(result.loa_upper, 73.9, places=1)
        self.assertLess(result.bias_ci[0], result.bias)
        self.assertGreater(result.bias_ci[1], result.bias)
        self.assertLess(result.loa_upper_ci[0], result.loa_upper)
        self.assertGreater(result.loa_upper_ci[1], result.loa_upper)

    def test_constructed_example_with_hand_computed_results(self):
        a = [10.0, 12.0, 14.0, 20.0]
        b = [9.0, 12.0, 17.0, 18.0]
        result = stats.bland_altman(a, b)
        # differences 1, 0, -3, 2: mean 0, variance 14/3
        self.assertAlmostEqual(result.bias, 0.0)
        self.assertAlmostEqual(result.sd_diff, math.sqrt(14 / 3))
        self.assertAlmostEqual(
            result.loa_upper, 1.959964 * math.sqrt(14 / 3), places=5)
        self.assertAlmostEqual(result.loa_lower, -result.loa_upper, places=9)
        t3 = 3.182446
        self.assertAlmostEqual(
            result.bias_ci[1], t3 * math.sqrt(14 / 3) / 2, places=4)
        se_loa = math.sqrt((14 / 3) * (1 / 4 + 1.959964 ** 2 / 6))
        self.assertAlmostEqual(
            result.loa_upper_ci[1], result.loa_upper + t3 * se_loa, places=4)

    def test_detects_proportional_bias(self):
        b = np.linspace(10, 100, 20)
        a = b * 1.2          # difference grows with magnitude
        result = stats.bland_altman(a, b)
        self.assertLess(result.slope_p_value, 1e-6)
        self.assertAlmostEqual(result.slope, 0.2 / 1.1, places=9)

    def test_constant_offset_has_no_slope_and_zero_sd(self):
        b = np.arange(1.0, 9.0)
        result = stats.bland_altman(b + 2.0, b)
        self.assertAlmostEqual(result.bias, 2.0)
        self.assertAlmostEqual(result.sd_diff, 0.0)
        self.assertAlmostEqual(result.slope, 0.0)

    def test_small_samples_are_nan_not_errors(self):
        self.assertTrue(math.isnan(stats.bland_altman([1.0], [2.0]).sd_diff))
        empty = stats.bland_altman([], [])
        self.assertEqual(empty.n, 0)
        self.assertTrue(math.isnan(empty.bias))
        with self.assertRaises(ValueError):
            stats.bland_altman([1, 2], [1])


class CorrelationTests(unittest.TestCase):
    def test_pearson_reference(self):
        result = stats.pearson([1, 2, 3, 4, 5], [2, 4, 5, 4, 5])
        self.assertAlmostEqual(result.r, 0.7745967, places=6)
        self.assertAlmostEqual(result.p_value, 0.124027, places=5)
        # Fisher-z interval computed by hand.
        z = math.atanh(result.r)
        se = 1 / math.sqrt(2)
        self.assertAlmostEqual(
            result.ci_low, math.tanh(z - 1.959964 * se), places=5)
        self.assertAlmostEqual(
            result.ci_high, math.tanh(z + 1.959964 * se), places=5)
        self.assertEqual(result.n, 5)

    def test_spearman_iq_tv_reference(self):
        iq = [86, 97, 99, 100, 101, 103, 106, 110, 112, 113]
        tv = [0, 20, 28, 27, 50, 29, 7, 17, 6, 12]
        result = stats.spearman(iq, tv)
        self.assertAlmostEqual(result.r, -0.1757576, places=6)
        self.assertAlmostEqual(result.p_value, 0.6272, places=3)

    def test_spearman_uses_average_ranks_for_ties(self):
        ranks = stats._average_ranks(np.array([10.0, 20.0, 20.0, 30.0, 20.0]))
        self.assertEqual(ranks.tolist(), [1.0, 3.0, 3.0, 5.0, 3.0])
        # A monotone but non-linear relation has rho 1, Pearson < 1.
        x = [1, 2, 3, 4, 5, 6]
        y = [1, 4, 9, 16, 25, 36]
        self.assertAlmostEqual(stats.spearman(x, y).r, 1.0)
        self.assertLess(stats.pearson(x, y).r, 1.0)

    def test_perfect_and_degenerate_correlations(self):
        perfect = stats.pearson([1, 2, 3, 4], [2, 4, 6, 8])
        self.assertAlmostEqual(perfect.r, 1.0)
        self.assertEqual(perfect.p_value, 0.0)
        self.assertTrue(math.isnan(stats.pearson([1, 1, 1], [1, 2, 3]).r))
        self.assertTrue(math.isnan(stats.pearson([1, 2], [1, 2]).r))
        self.assertTrue(math.isnan(stats.pearson([], []).r))
        with self.assertRaises(ValueError):
            stats.pearson([1, 2, 3], [1, 2])


class ErrorMetricTests(unittest.TestCase):
    def test_rmse_mae(self):
        a, b = [1.0, 2.0, 3.0], [1.0, 4.0, 6.0]
        self.assertAlmostEqual(stats.mae(a, b), 5 / 3)
        self.assertAlmostEqual(stats.rmse(a, b), math.sqrt(13 / 3))
        self.assertTrue(math.isnan(stats.rmse([], [])))
        self.assertTrue(math.isnan(stats.mae([], [])))

    def test_mean_absolute_percentage_difference(self):
        a, b = [10.0, 20.0], [12.0, 20.0]
        # |10-12| / 11 = 18.18 %, 0 %
        self.assertAlmostEqual(
            stats.mean_absolute_percentage_difference(a, b),
            (2 / 11 * 100 + 0.0) / 2)
        self.assertAlmostEqual(
            stats.mean_absolute_percentage_difference(a, b, denominator="b"),
            (2 / 12 * 100 + 0.0) / 2)
        # Pairs with a zero denominator are skipped.
        self.assertAlmostEqual(
            stats.mean_absolute_percentage_difference([0.0, 10.0], [0.0, 5.0]),
            5 / 7.5 * 100)
        self.assertTrue(math.isnan(
            stats.mean_absolute_percentage_difference([0.0], [0.0])))
        with self.assertRaises(ValueError):
            stats.mean_absolute_percentage_difference([1], [1], "x")

    def test_percent_agreement(self):
        self.assertAlmostEqual(stats.percent_agreement("abca", "abcb"), 75.0)
        self.assertTrue(math.isnan(stats.percent_agreement([], [])))


if __name__ == "__main__":
    unittest.main()
