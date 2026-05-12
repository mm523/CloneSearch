'''
Key CloneSearch functions. 

CloneSearch finds TCR frequency trajectories that look unusual - the ``outliers".
CloneSearch_clustering takes frequency trajectories and clusters them.

'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import optimal_leaf_ordering, leaves_list

from clonesearch.get_params import get_sigma_and_b
from clonesearch.normalisation_functions import noise_by_size_norm, log10_transform
from clonesearch.gaussian_outliers import find_gaussian_outliers


def pca_outlier_identification(normed_array, qc_clones, pval_or_fdr, statistical_threshold):
    '''
    Calculate the PCA. 

    normed_array = PCA input. Either g(f) or log10-transformed frequencies.
        In both cases, we expect these to be normalised by the maximum.
    qc_clones = list of clones, in the same order as frequencies in normed_array
    pval_or_fdr = how to calculate the threshold
    statistical_threshold = p-value or FDR threshold
    '''
    pca = PCA(whiten = True)
    pca_fit = pca.fit_transform(normed_array)

    if pval_or_fdr.lower() == 'pval':
        use_FDR = False
    elif pval_or_fdr.lower() == 'fdr':
        use_FDR = True

    R, outlier_vector, R_thresh = \
        find_gaussian_outliers(
            pca_fit,
            statistical_threshold=statistical_threshold,
            use_FDR=use_FDR
            )

    print('Number of outliers: ', sum(outlier_vector))
    pca_fit = pd.DataFrame(pca_fit, index = qc_clones)
    pca_fit['radius'] = R

    return pca_fit, R_thresh, outlier_vector

def CloneSearch(X_counts,
                N_r,
                all_clones,
                sample_order,
                tp_dict,
                statistical_threshold = .05,
                pval_or_fdr = 'fdr',
                which_beta = 'constantBeta',
                which_QC = 'strictQC',
                which_transform = 'g(f)'
                ):
    '''
    Calculation of outliers, starting from a table of counts.

    X_counts = MxN array of counts of M TCRs in N timepoints
    N_r = Nr for each sample in the timeseries (vector of length N)
    all_clones = list of clones of length M
    sample_order = list of timepoints of length N
    tp_dict = dictionary of format {timepoint_name:time_of_sampling} 
    statistical_threshold = FDR or pval threshold to use
    pval_or_FDR = whether to use a pval or a FDR threshold - alternatives: pval or fdr
    which_beta = use a constant beta or b parameter in g(f) for all samples in timeseries 
                    - alternatives: constantB, constantBeta. We recommend the constantBeta setting
    which_QC = include small clones or not - alternatives: looseQC, strictQC, noQC
    which_transform = use default g(f) transform or log10 - alternatives: g(f), log10

    RETURNS: 

    - list of outlier clones identified
    - PCA-transformed frequencies for each clone indicating the radius of each clone
    - radius threshold used
    - frequencies transformed by g(f)
    '''

    if which_QC == 'strictQC':
        # clones that are present with count >=3 at more than one timepoint
        mask = (X_counts > 2).sum(axis=1) > 1
    elif which_QC == 'looseQC':
        # this QC allows me to get more of the small clones
        # clones that are present at more than one timepoint
        mask = (X_counts > 0).sum(axis=1) > 1
    elif which_QC =='noQC':
        # all clones - assume user wants to run on everything
        mask = (X_counts > 0).sum(axis=1) > 0
    else:
        raise ValueError(
            'The parameter which_QC has an unrecognised value. '\
            'Please choose one of [strictQC, looseQC, noQC]'
            )

    freqs_all = X_counts/N_r
    freqs_qc = freqs_all[mask,:]
    counts_qc = X_counts[mask,:]
    qc_clones = np.array(all_clones)[mask]

    freq_info = pd.DataFrame(freqs_qc, index = qc_clones, columns=sample_order)

    if which_transform == 'g(f)':
        sigma, fit_b = get_sigma_and_b(freq_info, sample_order, tp_dict)
        fit_b = max(fit_b, (1/N_r).min())
        beta_factor = N_r*fit_b
        if which_beta == 'constantBeta':
            # this assumes that the noise factor is constant across samples,
            # and that the actual noise will depend on sample size
            beta_factor = [np.mean(beta_factor)]*len(beta_factor)

        # now transform using g(f)
        X_transformed = noise_by_size_norm(freqs_qc, N_r, sigma, beta_factor)
    elif which_transform == 'log10':
        X_transformed = log10_transform(counts_qc)
    else:
        raise ValueError(f'which_transform = "{which_transform}" not implemeted')

    X_transformed_norm = X_transformed - X_transformed.max(axis=1).reshape(-1, 1)
    pca_fit, R_thresh, outlier_vector = \
        pca_outlier_identification(
            X_transformed_norm, qc_clones,
            pval_or_fdr, statistical_threshold
            )
    outlier_list = qc_clones[outlier_vector]

    return outlier_list, pca_fit, R_thresh, X_transformed

def CloneSearch_clustering(input_df,
                           distance_metric = 'correlation',
                           linkage_method = 'average',
                           distance_thresh = 0.61):
    '''
    Clustering of a custom set of TCR frequencies, typically the outliers.

    input_df is what we are clustering on. 
        input_df of size MxN with M TCRs to cluster in N timepoints or PCA dimensions.
        For distance_metric = 'correlation', transformed and normalised frequencies are recommended. 
        For 'cosine' the PCA transformed data is recommended.
    distance_metric is passed as argument to pdist. 
    linkage_method is passed as argument
    We find distance_metric = 'correlation' and linkage_method = 'average' 
        to be a good combination for transformed frequencies and 
        distance_metric = 'cosine' and linkage_method = 'complete' 
        to be a good combination for PCA-transformed data
    '''

    _similarity_all = pdist(input_df, metric=distance_metric)

    row_linkage_all = linkage(
        _similarity_all, method=linkage_method
    )

    row_linkage_all = optimal_leaf_ordering(row_linkage_all, _similarity_all)

    fl = fcluster(row_linkage_all, distance_thresh, criterion='distance')

    def stable_cluster_labels(fl, row_linkage):
        dendrogram_order = leaves_list(row_linkage)
        remap = {}
        counter = 1
        for idx in dendrogram_order:
            cluster_id = fl[idx]
            if cluster_id not in remap:
                remap[cluster_id] = counter
                counter += 1
        return np.array([remap[c] for c in fl])

    fl = stable_cluster_labels(fl, row_linkage_all)
    return row_linkage_all, fl
