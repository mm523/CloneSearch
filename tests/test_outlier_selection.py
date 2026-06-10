'''
Test that selected outliers stay as expected by running end-to-end
'''
import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from clonesearch.load_data import load_data
from clonesearch.CloneSearch import CloneSearch

@pytest.fixture(scope='class')
def clone_search_inputs():
    metadata = pd.read_csv('test_data/input/metadata.txt')
    input_path = Path('test_data/input/')
    output_path = Path('test_data/output/')
    columns = ['bestVGene', 'bestJGene', 'clonalSequence', 'cloneCount', 'aaSeqCDR3']
    clone_id_cols = ['clonalSequence', 'bestVGene', 'bestJGene']
    sample_list = metadata['sample'].tolist()

    counts_df = load_data(input_path, output_path, 'tab', columns, clone_id_cols, sample_list)

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
    @pytest.mark.skipif(
        not os.path.exists('test_data/expected_output/outliers.txt'),
        reason="test data file not available"
    )
    def test_cloneserch_overall(self, clone_search_inputs):
        expected_outliers = pd.read_csv(
            'test_data/expected_output/outliers.txt', sep='\t', header=None
        )[0].tolist()

        outlier_list, pca_fit, R_thresh, X_transformed = CloneSearch(
            clone_search_inputs['counts'], clone_search_inputs['Nr'],
            clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
            clone_search_inputs['timepoint_dictionary'],
            0.05, 'fdr', 'constantBeta', 'strictQC', 'g'
        )
        assert len(outlier_list) == len(expected_outliers)
        assert sorted(outlier_list) == sorted(expected_outliers)

    def test_cloneserch_pval_does_not_crash(self, clone_search_inputs):
        outlier_list, pca_fit, R_thresh, X_transformed = CloneSearch(
            clone_search_inputs['counts'], clone_search_inputs['Nr'],
            clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
            clone_search_inputs['timepoint_dictionary'],
            0.05, 'pvalue', 'constantBeta', 'strictQC', 'g'
        )
        assert isinstance(R_thresh, float)
        assert isinstance(outlier_list, np.ndarray)

    def test_invalid_pval_or_fdr_raises_value_error(self, clone_search_inputs):
        with pytest.raises(ValueError, match='not recognised'):
            CloneSearch(
                clone_search_inputs['counts'], clone_search_inputs['Nr'],
                clone_search_inputs['all_clones'], clone_search_inputs['sample_order'],
                clone_search_inputs['timepoint_dictionary'],
                0.05, 'pval', 'constantBeta', 'strictQC', 'g'
            )
