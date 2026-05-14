'''
Test that behaviour of g(f) is as expected
'''

import numpy as np
from clonesearch.utils import normalisation_functions as norm

class TestNewTransform:
    def test_get_g_of_f(self):

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
