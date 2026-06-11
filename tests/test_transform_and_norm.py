'''
Test that behaviour of g(f) is as expected
'''

import numpy as np
import pytest
from clonesearch.utils import normalisation_functions as norm

class TestNewTransform:
    def test_get_g_of_f(self):
        # numerical testing

        f = 0.1
        Nr = 5000
        beta = 5
        sigma = 0.5
        g_of_1 = 27.63501512

        np.testing.assert_almost_equal(
            norm.g_of_f(f, Nr, sigma, beta), 23.06526392 - g_of_1,
            decimal = 5
            )

        f = 0.01
        Nr = 12458
        beta = 2.8
        sigma = 0.3
        g_of_1 = 52.60198
        np.testing.assert_almost_equal(
            norm.g_of_f(f, Nr, sigma, beta), 37.62914536 - g_of_1,
            decimal = 5
            )

        f = 0
        Nr = 12458
        beta = 2.8
        sigma = 0.3
        g_of_1 = 52.60198
        np.testing.assert_almost_equal(
            norm.g_of_f(f, Nr, sigma, beta), 28.00166283 - g_of_1,
            decimal = 5
            )
    
    def test_g_of_1_is_zero(self):
        result = norm.g_of_f(1.0, 1000, 0.5, 0.01)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_g_of_1_is_zero_for_array_input(self):
        f = np.ones(10)
        result = norm.g_of_f(f, 5000, 0.3, 0.005)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_g_of_1_is_zero_across_parameter_values(self):
        params = [
            (100,   0.1, 0.001),
            (1000,  0.5, 0.01),
            (50000, 0.9, 0.1),
        ]
        for N_r, s, beta in params:
            result = norm.g_of_f(1.0, N_r, s, beta)
            assert result == pytest.approx(0.0, abs=1e-10), (
                f'g(1) != 0 for N_r={N_r}, s={s}, beta={beta}'
            )
    
    def test_g_is_negative_below_one(self):
        f_values = np.array([0.001, 0.01, 0.1, 0.5, 0.9])
        result = norm.g_of_f(f_values, 10000, 0.5, 0.01)
        assert np.all(result < 0)

    def test_g_is_monotonically_increasing(self):
        f_values = np.linspace(0.001, 1.0, 200)
        result = norm.g_of_f(f_values, 10000, 0.5, 0.01)
        diffs = np.diff(result)
        assert np.all(diffs > 0), 'g(f) should be strictly increasing in f'
    
    def test_g_of_zero_is_finite_and_negative(self):
        result = norm.g_of_f(0.0, 12458, 0.3, 2.8)
        assert np.isfinite(result)
        assert result < 0
    
    def test_g_only_depends_on_beta_over_Nr_ratio(self):
        f = np.array([0.01, 0.1, 0.5])
        g1 = norm.g_of_f(f, 1000,  0.5, 1.0)
        g2 = norm.g_of_f(f, 10000, 0.5, 10.0)
        np.testing.assert_allclose(g1, g2, rtol=1e-8)
    
    def test_output_shape_matches_input(self):
        f = np.random.default_rng(0).uniform(0.001, 1.0, size=(30,))
        result = norm.g_of_f(f, 5000, 0.4, 0.01)
        assert result.shape == f.shape

    def test_no_nans_or_infs_for_valid_input(self):
        f = np.linspace(0.001, 0.999, 100)
        result = norm.g_of_f(f, 10000, 0.5, 0.01)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

class TestLog10Transform:

    def test_zero_counts_do_not_produce_neg_inf(self):
        '''Zero entries get pseudocount 1/3, so log should never produce -inf.'''
        X = np.array([[0, 2], [3, 0]], dtype=float)
        result = norm.log10_transform(X)
        assert not np.any(np.isneginf(result))

    def test_no_nans_or_infs_with_zeros_present(self):
        rng = np.random.default_rng(1)
        X = rng.integers(0, 100, size=(20, 5)).astype(float)
        X[X < 10] = 0
        result = norm.log10_transform(X)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_output_shape_matches_input(self):
        rng = np.random.default_rng(2)
        X = rng.integers(1, 50, size=(15, 4)).astype(float)
        result = norm.log10_transform(X)
        assert result.shape == X.shape

    def test_all_values_are_non_positive(self):
        '''log10 of a frequency (which lies in (0, 1]) must be <= 0.'''
        rng = np.random.default_rng(3)
        X = rng.integers(1, 100, size=(10, 3)).astype(float)
        result = norm.log10_transform(X)
        assert np.all(result <= 0)

    def test_larger_counts_give_larger_log_frequency(self):
        '''Within a sample, higher-count clones should have higher (less negative) log-frequency.'''
        X = np.array([[10], [1]], dtype=float)
        result = norm.log10_transform(X)
        assert result[0, 0] > result[1, 0]

class TestNoiseBySizeNorm:

    def test_noise_by_size_norm(self):

        Nr = np.array([12458, 7512, 74081])
        beta = np.array([2.8, 2.8, 2.8])
        sigma = 0.3

        g_of_1_1 = 52.60198
        g_of_1_2 = 49.23230504
        g_of_1_3 = 64.48382867

        f_array = np.array([
            [0.1, 0, 2.4*1e-4],
            [0, 2.4*1e-4, 0.1],
            [2.4*1e-4, 0.1, 0],
            [0.3, 0.056, 0.002]
        ])

        expected_output = np.array([
            [44.9637747 - g_of_1_1, 26.3154587 - g_of_1_2, 38.5970617 - g_of_1_3],
            [28.0016628 - g_of_1_1, 27.9051992 - g_of_1_2, 56.8148336 - g_of_1_3],
            [30.0366299 - g_of_1_1, 41.6181069 - g_of_1_2, 33.9443165 - g_of_1_3],
            [48.5984095 - g_of_1_1, 39.7373764 - g_of_1_2, 44.0930080 - g_of_1_3]
        ])

        X = norm.noise_by_size_norm(f_array, Nr, sigma, beta)

        np.testing.assert_array_almost_equal(expected_output, X)
    
    def test_output_shape_matches_input(self):
        rng = np.random.default_rng(42)
        df = rng.uniform(0.001, 0.1, size=(50, 4))
        N_r = np.array([10000, 8000, 12000, 9000])
        beta = np.array([0.01, 0.01, 0.01, 0.01])
        result = norm.noise_by_size_norm(df, N_r, 0.5, beta)
        assert result.shape == df.shape

    def test_no_nans_or_infs(self):
        rng = np.random.default_rng(42)
        df = rng.uniform(0.001, 0.5, size=(30, 3))
        N_r = np.array([10000, 8000, 12000])
        beta = np.array([0.01, 0.01, 0.01])
        result = norm.noise_by_size_norm(df, N_r, 0.5, beta)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_consistent_with_g_of_f_column_by_column(self):
        rng = np.random.default_rng(7)
        df = rng.uniform(0.001, 0.5, size=(10, 3))
        N_r = np.array([10000, 8000, 12000])
        beta = np.array([0.01, 0.015, 0.008])
        s = 0.4

        result = norm.noise_by_size_norm(df, N_r, s, beta)

        for i in range(3):
            expected = norm.g_of_f(df[:, i], N_r[i], s, beta[i])
            np.testing.assert_allclose(result[:, i], expected, rtol=1e-10)
