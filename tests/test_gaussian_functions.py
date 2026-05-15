'''
Test functions to find outliers
'''
import numpy as np
from clonesearch.utils import gaussian_outliers as gauss

class TestRadiusFunctions:
    '''
    Calculated Pr is as expected
    '''
    def test_P_r(self):
        x = [0.1, 3, 5, 10, 15]
        n = 10

        Pr = np.array([
            2.59118E-12, 0.569422862, 0.018954738, 5.02279E-16, 1.38791E-41
        ])

        calculated_Pr = gauss.P_r(x, n)

        np.testing.assert_allclose(calculated_Pr, Pr, atol = 1e-6)

    def test_find_radius(self):
        pca_df = np.array([
            [0.1, 0, 0, 0, 0],
            [1, 3, 5, 10, 15],
            [0, 0, 3, 0, 0],
        ])

        expected_output = np.array(
            [0.1, np.sqrt(1+9+25+100+225), 3]
        )

        radii = gauss.find_radius(pca_df)

        np.testing.assert_allclose(radii, expected_output, atol = 1e-6)

class TestGaussOutlierDef:
    '''
    Outlier definition is as expected
    '''
    def test_find_gaussian_outliers(self):
        pca_df = np.array([
            [0.1, 0, 0, 0, 0],
            [1, 3, 5, 10, 15],
            [0, 0, 3, 0, 0],
        ])

        radii, outlier_vector, radius_thresh = \
            gauss.find_gaussian_outliers(pca_df, statistical_threshold=0.05, use_FDR=False)

        expected_radii = np.array(
            [0.1, np.sqrt(1+9+25+100+225), 3]
        )
        np.testing.assert_allclose(radii, expected_radii, atol = 1e-6)

        expected_radius_thresh = 3.327221063
        np.testing.assert_allclose(radius_thresh, expected_radius_thresh, atol = 1e-3)

        expected_outlier_vector = [False, True, False]
        np.testing.assert_array_equal(expected_outlier_vector, outlier_vector)
