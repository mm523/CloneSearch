import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from get_params import get_sigma_and_beta
from normalisation_functions import noise_by_size_norm
from gaussian_outliers import find_gaussian_outliers_fixed_sigma

def CloneSearch(X_counts, 
                N_r, 
                all_clones,
                sample_order,
                tp_dict,
                statistical_threshold = .05,
                pval_or_fdr = 'fdr',
                which_beta = 'constantBeta',
                which_QC = 'strictQC'
                ):
    '''
    X_counts = MxN array of counts - each row = TCR (total M TCRs); each column = timepoint (total N timepoints)

    N_r = Nr for each sample in the timeseries (vector of length N)

    all_clones = list of clones of length M

    sample_order = list of timepoints of length N

    tp_dict = dictionary of format {timepoint_name:time_of_sampling} 

    statistical_threshold = FDR or pval threshold to use

    pval_or_FDR = whether to use a pval or a FDR threshold - alternatives: pval or fdr

    which_beta = use a constant beta parameter or a constant B parameter in g(f) for all samples in timeseries - alternatives: constantB, constantBeta

    which_QC = include small clones or not - alternatives: looseQC, strictQC

    RETURNS: 

    - list of outlier clones identified
    - PCA-transformed frequencies for each clone indicating the radius of each clone
    - radius threshold used
    - frequencies transformed by g(f)

    '''

    if which_QC == 'strictQC':
        mask = (X_counts > 2).sum(axis=1) > 1 # clones that are present with count >=3 at more than one timepoint
    else:
        # this QC allows me to get more of the small clones
        mask = (X_counts > 0).sum(axis=1) > 1 # clones that are present at more than one timepoint

    freqs_all = X_counts/N_r

    freqs_qc = freqs_all[mask,:]
    qc_clones = np.array(all_clones)[mask]

    freq_info = pd.DataFrame(freqs_qc, index = qc_clones, columns=sample_order)

    sigma, fit_b = get_sigma_and_beta(freq_info, tp_dict)
    beta_factor = N_r*fit_b
    if which_beta == 'constantBeta':
        # this assumes that the noise factor is constant across samples, and that the actual noise will depend on sample size
        beta_factor = [np.mean(beta_factor)]*len(beta_factor)

    # now transform using g(f)
    X_transformed = noise_by_size_norm(freqs_qc, N_r, sigma, beta_factor)
    X_transformed_norm = X_transformed - X_transformed.max(axis=1).reshape(-1, 1)

    # calculate the PCA
    pca = PCA(whiten = True)
    pca_fit = pca.fit_transform(X_transformed_norm)

    if pval_or_fdr == 'pval':
        use_FDR = False
    elif pval_or_fdr == 'fdr':
        use_FDR = True

    R, outlier_vector, R_thresh = find_gaussian_outliers_fixed_sigma(pca_fit, statistical_threshold=statistical_threshold, use_FDR=use_FDR)

    print('Number of outliers: ', sum(outlier_vector))

    outlier_list = qc_clones[outlier_vector]
    pca_fit['radius'] = R

    return outlier_list, pca_fit, R_thresh, X_transformed





    
