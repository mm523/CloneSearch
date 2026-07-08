'''
Test functions to find outliers
'''
import numpy as np
import pytest
from scipy.integrate import quad
from clonesearch.utils.gaussian_outliers import (
    find_radius,
    find_radius_outlier_thresh_pval,
    find_fdr_thresh,
    find_gaussian_outliers,
    theoretical_cdf,
    P_r,
)

class TestFindRadius:

    def test_hardcoded_known_vectors(self):
        pca_df = np.array([
            [0.1, 0, 0, 0, 0],
            [1, 3, 5, 10, 15],
            [0, 0, 3, 0, 0],
        ])
        expected = np.array(
            [0.1, np.sqrt(1+9+25+100+225), 3]
        )
        np.testing.assert_allclose(find_radius(pca_df), expected, atol=1e-6)

    def test_known_3d_vector(self):
        '''Radius of [3, 4, 0] is 5 by Pythagoras.'''
        pca_df = np.array([[3.0, 4.0, 0.0]])
        r = find_radius(pca_df)
        np.testing.assert_allclose(r, [5.0])

    def test_origin_has_zero_radius(self):
        pca_df = np.array([[0.0, 0.0, 0.0]])
        r = find_radius(pca_df)
        np.testing.assert_allclose(r, [0.0])

    def test_output_length_matches_input_rows(self):
        rng = np.random.default_rng(0)
        pca_df = rng.standard_normal((25, 5))
        r = find_radius(pca_df)
        assert len(r) == 25

    def test_all_radii_non_negative(self):
        rng = np.random.default_rng(1)
        pca_df = rng.standard_normal((50, 4))
        r = find_radius(pca_df)
        assert np.all(r >= 0)

    def test_scaling_scales_radius(self):
        '''Doubling all coordinates should double the radius.'''
        pca_df = np.array([[1.0, 2.0, 3.0]])
        r_orig   = find_radius(pca_df)
        r_scaled = find_radius(2 * pca_df)
        np.testing.assert_allclose(r_scaled, 2 * r_orig)

class TestFindRadiusOutlierThreshPval:

    def test_threshold_satisfies_theoretical_survival(self):
        '''
        The returned threshold r should satisfy theoretical_cdf(r, dims) = 1 - pval,
        i.e. exactly pval probability mass lies above r under the null.
        Tested across multiple (dims, pval) combinations.
        '''
        for dims, pval in [(4, 0.05), (6, 0.01), (10, 0.025)]:
            r_thresh = find_radius_outlier_thresh_pval(dims, pval)
            cdf_at_thresh = theoretical_cdf(r_thresh, dims)
            assert cdf_at_thresh == pytest.approx(1 - pval, abs=1e-6), (
                f'CDF({r_thresh:.3f}) = {cdf_at_thresh:.6f}, expected {1 - pval} '
                f'(dims={dims}, pval={pval})'
            )

    def test_larger_pval_gives_smaller_threshold(self):
        '''A more lenient p-value should correspond to a smaller radius threshold.'''
        dims = 4
        r_strict  = find_radius_outlier_thresh_pval(dims, pval=0.01)
        r_lenient = find_radius_outlier_thresh_pval(dims, pval=0.10)
        assert r_strict > r_lenient

    def test_more_dimensions_gives_larger_threshold(self):
        '''The expected radius grows with dimensionality, so the threshold should too.'''
        pval  = 0.05
        r_low  = find_radius_outlier_thresh_pval(dimensions=2,  pval=pval)
        r_high = find_radius_outlier_thresh_pval(dimensions=10, pval=pval)
        assert r_high > r_low

    def test_returns_positive_float(self):
        r = find_radius_outlier_thresh_pval(dimensions=5, pval=0.05)
        assert isinstance(r, float)
        assert r > 0

class TestFindFdrThresh:

    def test_returns_a_float(self):
        rng = np.random.default_rng(0)
        radii = np.sqrt(rng.chisquare(df=4, size=200))
        thresh = find_fdr_thresh(radii, dims=4, fdr=0.05)
        assert isinstance(thresh, float)

    def test_null_data_threshold_equals_maximum(self):
        '''
        For purely null data, the FDR criterion is never met, idx=-1 fires,
        and the threshold equals the maximum observed radius.
        '''
        rng = np.random.default_rng(42)
        radii = np.sqrt(rng.chisquare(df=4, size=500))
        thresh = find_fdr_thresh(radii, dims=4, fdr=0.05)
        assert thresh == pytest.approx(np.max(radii))

    def test_obvious_outliers_exceed_threshold(self):
        '''
        With 20 very large radii injected into a null dataset, the threshold
        should fall below all injected values so every outlier is captured.
        '''
        rng = np.random.default_rng(7)
        null_radii    = np.sqrt(rng.chisquare(df=4, size=300))
        outlier_radii = rng.uniform(20, 25, size=20)
        radii = np.concatenate([null_radii, outlier_radii])
        thresh = find_fdr_thresh(radii, dims=4, fdr=0.05)
        assert thresh <= np.min(outlier_radii)

    def test_single_unique_radius_triggers_fallback(self):
        '''
        All identical radii → division by zero in the ratio → IndexError fallback.
        The returned threshold should equal the only observed value.
        '''
        radii = np.full(50, 3.0)
        thresh = find_fdr_thresh(radii, dims=4, fdr=0.05)
        assert thresh == pytest.approx(3.0)

    def test_threshold_within_range_of_radii(self):
        rng = np.random.default_rng(99)
        radii = np.sqrt(rng.chisquare(df=6, size=100))
        thresh = find_fdr_thresh(radii, dims=6, fdr=0.05)
        assert np.min(radii) <= thresh <= np.max(radii)

class TestFindGaussianOutliers:

    def test_hardcoded_golden_values(self):
        '''
        Golden-value regression: pins the exact threshold and outlier vector
        for a known 3-row input. Catches unintended numerical changes to the
        chi-distribution solver or threshold logic.
        '''
        pca_df = np.array([
            [0.1, 0, 0, 0, 0],
            [1, 3, 5, 10, 15],
            [0, 0, 3, 0, 0],
        ])
        radii, outlier_vector, radius_thresh = find_gaussian_outliers(
            pca_df, statistical_threshold=0.05, use_FDR=False
        )
        expected_radii = np.array(
            [0.1, np.sqrt(1+9+25+100+225), 3]
        )
        np.testing.assert_allclose(radii, expected_radii, atol=1e-6)
        np.testing.assert_allclose(radius_thresh, 3.327221063, atol=1e-3)
        np.testing.assert_array_equal(outlier_vector, [False, True, False])

    def test_output_shapes(self):
        rng = np.random.default_rng(0)
        pca_fit = rng.standard_normal((40, 4))
        radii, outlier_vector, R_thresh = find_gaussian_outliers(pca_fit)
        assert len(radii) == 40
        assert len(outlier_vector) == 40
        assert isinstance(R_thresh, float)

    def test_outlier_vector_is_boolean(self):
        rng = np.random.default_rng(1)
        pca_fit = rng.standard_normal((30, 3))
        radii, outlier_vector, R_thresh = find_gaussian_outliers(pca_fit)
        assert outlier_vector.dtype == bool

    def test_outlier_vector_consistent_with_threshold(self):
        '''Flagged clones must have radius > R_thresh; non-flagged must have radius <= R_thresh.'''
        rng = np.random.default_rng(5)
        pca_fit = rng.standard_normal((50, 4))
        radii, outlier_vector, R_thresh = find_gaussian_outliers(pca_fit)
        assert np.all(radii[outlier_vector]  > R_thresh)
        assert np.all(radii[~outlier_vector] <= R_thresh)

    def test_fdr_and_pval_modes_give_different_thresholds(self):
        '''
        FDR and p-value thresholding are distinct methods and should
        produce different thresholds on the same data.
        Uses data with injected outliers so both methods have signal to work with.
        '''
        rng = np.random.default_rng(10)
        null    = rng.standard_normal((90, 5))
        outlier = rng.standard_normal((10, 5)) + 5
        pca_fit = np.vstack([null, outlier])
        _, _, thresh_fdr  = find_gaussian_outliers(pca_fit, use_FDR=True)
        _, _, thresh_pval = find_gaussian_outliers(pca_fit, use_FDR=False)
        assert thresh_fdr != pytest.approx(thresh_pval, rel=0.01)

class TestTheoreticalCdf:

    def test_cdf_at_zero_is_zero(self):
        assert theoretical_cdf(0.0, n=4) == pytest.approx(0.0)

    def test_cdf_approaches_one_at_large_radius(self):
        assert theoretical_cdf(100.0, n=4) == pytest.approx(1.0, abs=1e-6)

    def test_cdf_is_monotonically_increasing(self):
        w = np.linspace(0.1, 6, 100) # stop at 6 to account for numerical saturation
        cdf = theoretical_cdf(w, n=4)
        assert np.all(np.diff(cdf) > 0)

    def test_consistent_with_pval_threshold(self):
        '''
        theoretical_cdf(find_radius_outlier_thresh_pval(dims, pval), dims) should
        equal 1 - pval — a cross-module consistency check.
        '''
        dims, pval = 8, 0.05
        r_thresh = find_radius_outlier_thresh_pval(dims, pval)
        assert theoretical_cdf(r_thresh, dims) == pytest.approx(1 - pval, abs=1e-6)

class TestPr:

    def test_hardcoded_spot_check(self):
        '''Regression anchor: P_r values for x=[0.1,3,5,10,15], n=10.'''
        x  = [0.1, 3, 5, 10, 15]
        expected = np.array([
            2.59118e-12, 0.569422862, 0.018954738, 5.02279e-16, 1.38791e-41
        ])
        np.testing.assert_allclose(P_r(x, 10), expected, atol=1e-6)

    def test_integrates_to_one(self):
        '''P_r is a probability density — it must integrate to 1 over [0, inf).'''
        for n in [2, 4, 6, 10]:
            integral, _ = quad(lambda r: P_r(r, n), 0, np.inf)
            assert integral == pytest.approx(1.0, abs=1e-4), (
                f'P_r integrates to {integral:.6f} for n={n}, expected 1.0'
            )

    def test_non_negative(self):
        r_vals = np.linspace(0.01, 10, 100)
        for n in [3, 5, 8]:
            result = P_r(r_vals, n)
            assert np.all(result >= 0)

    def test_mode_increases_with_dimensions(self):
        '''
        The mode of the chi distribution grows with the number of dimensions.
        Higher-dimensional PCA spaces → larger expected radii under the null.
        '''
        r_vals    = np.linspace(0.01, 15, 500)
        mode_low  = r_vals[np.argmax(P_r(r_vals, n=2))]
        mode_high = r_vals[np.argmax(P_r(r_vals, n=10))]
        assert mode_high > mode_low