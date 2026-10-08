'''
Test that selected outliers stay as expected by running end-to-end,
plus unit tests for the refactored building blocks (qc_transform,
compute_qc_sweep_stats, select_best_qc, choose_best_clone_QC,
CloneSearch_clustering / stable cluster labelling).
'''
import os
import matplotlib
matplotlib.use('Agg')  # never open a GUI window / block during tests

from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from clonesearch.load_data import load_data
from clonesearch.CloneSearch import (
    CloneSearch,
    CloneSearch_clustering,
    choose_best_clone_QC,
    compute_qc_sweep_stats,
    select_best_qc,
    qc_transform,
    _stable_cluster_labels,
)


@pytest.fixture(autouse=True)
def _no_plt_show(monkeypatch):
    '''
    Belt-and-braces: even with the Agg backend, don't let any test
    actually block on plt.show(). Applies to every test in this module.
    '''
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, 'show', lambda *args, **kwargs: None)


@pytest.fixture(scope='class')
def clone_search_inputs():
    metadata = pd.read_csv('test_data/input/metadata.txt')
    input_path = Path('test_data/input/')
    output_path = Path('test_data/output/')
    columns = ['bestVGene', 'bestJGene', 'clonalSequence', 'cloneCount', 'aaSeqCDR3']
    clone_id_cols = ['clonalSequence', 'bestVGene', 'bestJGene']
    sample_list = metadata['sample'].tolist()

    counts_df = load_data(input_path, 'tab', columns, clone_id_cols, sample_list)

    sample_order = metadata.sort_values(by='timepoint', ascending=True)['sample'].tolist()
    timepoint_dictionary = dict(zip(metadata['sample'].tolist(), metadata['timepoint'].tolist()))

    return dict(
        counts=counts_df.values,
        Nr=counts_df.sum(axis=0).values,
        all_clones=counts_df.index.tolist(),
        sample_order=sample_order,
        timepoint_dictionary=timepoint_dictionary,
    )

class TestCloneSearch:
    def test_cloneserch_overall(self, clone_search_inputs):
        expected_outliers = pd.read_csv(
            'test_data/expected_output/outliers_BH.txt', sep='\t', header=None
        )[0].tolist()

        outlier_list, pca_fit, R_thresh, X_transformed = CloneSearch(
            clone_search_inputs['counts'], clone_search_inputs['Nr'],
            clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
            clone_search_inputs['timepoint_dictionary'],
            0.05, 'fdr', 'constantBeta', 'strict', 'g'
        )
        assert len(outlier_list) == len(expected_outliers)
        assert sorted(outlier_list) == sorted(expected_outliers)

    def test_cloneserch_overall_legacy(self, clone_search_inputs):
            expected_outliers = pd.read_csv(
                'test_data/expected_output/outliers_paper.txt', sep='\t', header=None
            )[0].tolist()
    
            outlier_list, pca_fit, R_thresh, X_transformed = CloneSearch(
                clone_search_inputs['counts'], clone_search_inputs['Nr'],
                clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
                clone_search_inputs['timepoint_dictionary'],
                0.05, 'fdr', 'constantBeta', 'strict', 'g', fdr_mode = 'legacy'
            )
            assert len(outlier_list) == len(expected_outliers)
            assert sorted(outlier_list) == sorted(expected_outliers)

    def test_cloneserch_pval_does_not_crash(self, clone_search_inputs):
        outlier_list, pca_fit, R_thresh, X_transformed = CloneSearch(
            clone_search_inputs['counts'], clone_search_inputs['Nr'],
            clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
            clone_search_inputs['timepoint_dictionary'],
            0.05, 'pvalue', 'constantBeta', 'strict', 'g',
            make_qc_plot=False,
        )
        assert isinstance(R_thresh, float)
        assert isinstance(outlier_list, np.ndarray)

    def test_invalid_pval_or_fdr_raises_value_error(self, clone_search_inputs):
        with pytest.raises(ValueError, match='not recognised'):
            CloneSearch(
                clone_search_inputs['counts'], clone_search_inputs['Nr'],
                clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
                clone_search_inputs['timepoint_dictionary'],
                0.05, 'pval', 'constantBeta', 'strict', 'g',
                make_qc_plot=False,
            )

class TestQCTransform:
    '''
    Unit tests for qc_transform, the helper shared between
    choose_best_clone_QC and CloneSearch.
    '''

    def test_qc_transform_shapes_and_normalisation(self, clone_search_inputs):
        clone_min = 0
        qc_mask = clone_search_inputs['counts'].sum(axis=1) > clone_min

        qc_clones, X_transformed, X_transformed_norm = qc_transform(
            clone_search_inputs['counts'], clone_search_inputs['Nr'],
            clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
            clone_search_inputs['timepoint_dictionary'],
            'g', 'constantBeta', qc_mask,
        )

        n_qc_clones = (clone_search_inputs['counts'].sum(axis=1) > clone_min).sum()
        assert len(qc_clones) == n_qc_clones
        assert X_transformed.shape == (n_qc_clones, len(clone_search_inputs['sample_order']))
        assert X_transformed_norm.shape == X_transformed.shape

        # each row should be normalised by its own max, so the row max is ~0
        row_maxes = X_transformed_norm.max(axis=1)
        assert np.allclose(row_maxes, 0.0, atol=1e-8)

    def test_qc_transform_stricter_threshold_keeps_fewer_or_equal_clones(self, clone_search_inputs):

        qc_mask = clone_search_inputs['counts'].sum(axis=1) > 0
        qc_clones_loose, _, _ = qc_transform(
            clone_search_inputs['counts'], clone_search_inputs['Nr'],
            clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
            clone_search_inputs['timepoint_dictionary'],
            'g', 'constantBeta', qc_mask,
        )

        qc_mask = clone_search_inputs['counts'].sum(axis=1) > 10
        qc_clones_strict, _, _ = qc_transform(
            clone_search_inputs['counts'], clone_search_inputs['Nr'],
            clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
            clone_search_inputs['timepoint_dictionary'],
            'g', 'constantBeta', qc_mask
        )
        assert len(qc_clones_strict) <= len(qc_clones_loose)
        # every clone kept under the stricter threshold must also be kept under the loose one
        assert set(qc_clones_strict).issubset(set(qc_clones_loose))

    def test_qc_transform_raises_on_invalid_transform(self, clone_search_inputs):
        qc_mask = clone_search_inputs['counts'].sum(axis=1) > 0
        with pytest.raises(ValueError, match='not implemented'):
            qc_transform(
                clone_search_inputs['counts'], clone_search_inputs['Nr'],
                clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
                clone_search_inputs['timepoint_dictionary'],
                'not_a_real_transform', 'constantBeta', qc_mask,
            )

    def test_qc_transform_raises_on_invalid_beta(self, clone_search_inputs):
        qc_mask = clone_search_inputs['counts'].sum(axis=1) > 0
        with pytest.raises(ValueError, match='not implemented'):
            qc_transform(
                clone_search_inputs['counts'], clone_search_inputs['Nr'],
                clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
                clone_search_inputs['timepoint_dictionary'],
                'g', 'not_a_real_beta', qc_mask,
            )


class TestQCSweep:
    '''
    Unit tests for compute_qc_sweep_stats / select_best_qc / choose_best_clone_QC.
    '''

    def test_compute_qc_sweep_stats_shape_and_columns(self, clone_search_inputs):
        clone_sums_to_sweep = [0, 1, 2]
        qc_stats = compute_qc_sweep_stats(
            clone_search_inputs['counts'][:10**4,:], clone_search_inputs['Nr'],
            clone_search_inputs['all_clones'][:10**4], clone_search_inputs['sample_order'],
            clone_search_inputs['timepoint_dictionary'],
            'g', 'constantBeta',
            clone_sums_to_sweep=clone_sums_to_sweep,
        )

        assert list(qc_stats['QC']) == clone_sums_to_sweep
        for col in ['QC', 'n_clones', 'ksstats', 'cmstats', 'andersonstats']:
            assert col in qc_stats.columns

        # goodness-of-fit statistics should be finite, non-negative numbers
        for col in ['ksstats', 'cmstats', 'andersonstats']:
            assert np.all(np.isfinite(qc_stats[col]))
            assert np.all(qc_stats[col] >= 0)

        # n_clones should shrink (or stay the same) as clone_min increases
        assert qc_stats['n_clones'].is_monotonic_decreasing

    def test_select_best_qc_picks_minimum(self):
        qc_stats = pd.DataFrame({
            'QC': [0, 1, 2, 3],
            'ksstats': [0.5, 0.2, 0.4, 0.3],
        })
        assert select_best_qc(qc_stats) == 1

    def test_select_best_qc_breaks_ties_with_smallest_qc(self):
        qc_stats = pd.DataFrame({
            'QC': [0, 1, 2, 3],
            'ksstats': [0.3, 0.1, 0.1, 0.5],
        })
        # QC=1 and QC=2 are tied for lowest ksstats; smallest QC should win
        assert select_best_qc(qc_stats) == 1

    def test_select_best_qc_tie_breaking_is_order_independent(self):
        # Same tie as above (QC=1 and QC=2 tied at 0.1), but rows shuffled so
        # the winning QC is NOT the first-encountered row. This guards against
        # a regression to "first match" logic instead of "smallest QC" logic.
        qc_stats = pd.DataFrame({
            'QC': [3, 2, 0, 1],
            'ksstats': [0.5, 0.1, 0.3, 0.1],
        })
        assert select_best_qc(qc_stats) == 1

    def test_select_best_qc_tie_at_largest_qc_in_sweep(self):
        # Tie involves the two largest QC values; smallest of the tied
        # values (2) should still be preferred over the untied smaller QC (0).
        qc_stats = pd.DataFrame({
            'QC': [0, 1, 2, 3],
            'ksstats': [0.4, 0.9, 0.1, 0.1],
        })
        assert select_best_qc(qc_stats) == 2

    def test_select_best_qc_three_way_tie_picks_smallest(self):
        qc_stats = pd.DataFrame({
            'QC': [5, 1, 9, 3],
            'ksstats': [0.2, 0.2, 0.2, 0.8],
        })
        # QC 5, 1, 9 are all tied at the minimum; smallest (1) must win
        assert select_best_qc(qc_stats) == 1

    def test_select_best_qc_tie_including_smallest_possible_qc(self):
        qc_stats = pd.DataFrame({
            'QC': [0, 4, 7],
            'ksstats': [0.15, 0.15, 0.9],
        })
        # tie between QC=0 and QC=4; QC=0 should win
        assert select_best_qc(qc_stats) == 0

    def test_select_best_qc_no_tie_is_unaffected(self):
        # sanity check that the tie-breaking path doesn't interfere with
        # the straightforward case of a single unambiguous minimum
        qc_stats = pd.DataFrame({
            'QC': [0, 1, 2, 3],
            'ksstats': [0.9, 0.05, 0.4, 0.6],
        })
        assert select_best_qc(qc_stats) == 1

    def test_select_best_qc_respects_stat_col_argument(self):
        qc_stats = pd.DataFrame({
            'QC': [0, 1, 2],
            'ksstats': [0.5, 0.1, 0.3],
            'cmstats': [0.1, 0.5, 0.3],
        })
        assert select_best_qc(qc_stats, stat_col='ksstats') == 1
        assert select_best_qc(qc_stats, stat_col='cmstats') == 0

    def test_choose_best_clone_QC_returns_a_swept_value(self, clone_search_inputs):
        chosen_QC = choose_best_clone_QC(
            clone_search_inputs['counts'], clone_search_inputs['Nr'],
            clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
            clone_search_inputs['timepoint_dictionary'],
            'g', 'constantBeta',
            make_plot=False,
        )
        expected_sweep = list(range(0, 15)) + [20]
        assert chosen_QC in expected_sweep

    def test_choose_best_clone_QC_make_plot_true_does_not_raise(self, clone_search_inputs):
        to_sweep = list(range(0, 5)) + [20]
        chosen_QC = choose_best_clone_QC(
            clone_search_inputs['counts'], clone_search_inputs['Nr'],
            clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
            clone_search_inputs['timepoint_dictionary'],
            'g', 'constantBeta',
            make_plot=True,
            clone_sums_to_sweep=to_sweep
        )
        assert isinstance(chosen_QC, (int, np.integer))


class TestCloneSearchClustering:
    '''
    Tests for CloneSearch_clustering and the stable cluster-labelling logic.
    '''

    @staticmethod
    def _two_well_separated_blobs():
        rng = np.random.default_rng(0)
        block_a = rng.normal(loc=0.0, scale=0.05, size=(5, 6))
        block_b = rng.normal(loc=5.0, scale=0.05, size=(5, 6))
        return pd.DataFrame(np.vstack([block_a, block_b]))

    def test_clustering_recovers_two_clusters(self):
        input_df = self._two_well_separated_blobs()
        row_linkage, labels = CloneSearch_clustering(
            input_df, distance_metric='euclidean', linkage_method='average',
            distance_thresh=2.0,
        )
        assert len(labels) == len(input_df)
        assert len(set(labels)) == 2
        # the two blocks (rows 0-4 and 5-9) should each be a single cluster
        assert len(set(labels[:5])) == 1
        assert len(set(labels[5:])) == 1
        assert labels[0] != labels[5]

    def test_clustering_labels_start_at_one_and_are_contiguous(self):
        input_df = self._two_well_separated_blobs()
        _, labels = CloneSearch_clustering(
            input_df, distance_metric='euclidean', linkage_method='average',
            distance_thresh=2.0,
        )
        unique_sorted = sorted(set(labels))
        assert unique_sorted == list(range(1, len(unique_sorted) + 1))

    def test_stable_cluster_labels_follow_dendrogram_leaf_order(self):
        from scipy.cluster.hierarchy import linkage as scipy_linkage

        # 4 points, two obvious pairs
        data = np.array([[0, 0], [0.1, 0], [10, 10], [10.1, 10]])
        row_linkage = scipy_linkage(data, method='average')
        # raw fcluster labels are arbitrary integers; simulate e.g. [2, 2, 1, 1]
        raw_fl = np.array([2, 2, 1, 1])

        relabelled = _stable_cluster_labels(raw_fl, row_linkage)

        # points 0 and 1 must share a label, points 2 and 3 must share a label,
        # and the two pairs must differ
        assert relabelled[0] == relabelled[1]
        assert relabelled[2] == relabelled[3]
        assert relabelled[0] != relabelled[2]
        # labels should be renumbered starting from 1
        assert set(relabelled) == {1, 2}