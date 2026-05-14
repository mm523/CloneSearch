from pathlib import Path
import pandas as pd
from clonesearch.load_data import load_data
from clonesearch.CloneSearch import CloneSearch

class TestCloneSearch:

    def test_cloneserch_overall(self):

        expected_outliers = pd.read_csv(
            'test_data/expected_output/outliers.txt', 
            sep = '\t', header=None
        )[0].tolist()

        metadata = pd.read_csv(
            'test_data/input/metadata.txt'
        )
        input_path = Path('test_data/input/')
        output_path = Path('test_data/output/')
        delimiter = 'tab'
        columns = [
            'bestVGene', 'bestJGene', 'clonalSequence',
            'cloneCount', 'aaSeqCDR3'
        ]
        clone_id_cols = ['clonalSequence','bestVGene', 'bestJGene']
        sample_list = metadata['sample'].tolist()

        counts_df = load_data(input_path, output_path, delimiter, columns, clone_id_cols, sample_list)

        sample_order = metadata.sort_values(by = 'timepoint', ascending=True)['sample'].tolist()
        timepoint_dictionary = dict(zip(metadata['sample'].tolist(), metadata['timepoint'].tolist()))
        all_clones = counts_df.index.tolist()
        Nr = counts_df.sum(axis=0).values
        counts = counts_df.values

        stat_thresh = 0.05
        pval_or_fdr = 'fdr'
        which_beta = 'constantBeta'
        which_QC = 'strictQC'
        which_transform = 'g(f)'

        outlier_list, pca_fit, R_thresh, X_transformed = \
                    CloneSearch(counts, Nr, all_clones,
                                sample_order, timepoint_dictionary, stat_thresh,
                                pval_or_fdr, which_beta, which_QC, which_transform)
        
        assert len(outlier_list) == len(expected_outliers)
        assert sorted(outlier_list) == sorted(expected_outliers)