'''
Key CloneSearch functions. 

CloneSearch finds TCR frequency trajectories that look unusual - the ``outliers".
CloneSearch_clustering takes frequency trajectories and clusters them.

'''

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.cluster.hierarchy import optimal_leaf_ordering, leaves_list
import matplotlib.pyplot as plt

from clonesearch.utils.get_params import get_sigma_and_b
from clonesearch.utils.normalisation_functions import noise_by_size_norm, log10_transform
from clonesearch.utils.gaussian_outliers import find_gaussian_outliers, find_radius

def anderson_darling_stat(data, cdf_func):
    x = np.sort(np.asarray(data))
    n = len(x)
    F = cdf_func(x)
    F = np.clip(F, 1e-10, 1 - 1e-10)  # avoid log(0) from CDF saturation
    i = np.arange(1, n + 1)
    S = np.sum((2 * i - 1) * (np.log(F) + np.log(1 - F[::-1])))
    return -n - S / n


def calculate_transformed_sequences(which_transform, which_beta, N_r, freq_info,
                                     freqs_qc, counts_qc, sample_order, tp_dict):
    if which_transform == 'g':
        sigma, fit_b = get_sigma_and_b(freq_info, sample_order, tp_dict)
        fit_b = max(fit_b, (1 / N_r).min())
        beta_factor = N_r * fit_b
        if which_beta == 'constantBeta':
            # assumes the noise factor is constant across samples, and that
            # the actual noise will depend on sample size
            beta_factor = [np.mean(beta_factor)] * len(beta_factor)
        elif which_beta == 'constantB':
            # b (rather than beta) is held constant across samples here;
            # beta_factor = N_r * fit_b is left to vary with sample size as-is
            pass
        else:
            raise ValueError(f'which_beta = "{which_beta}" not implemented')

        # now transform using g(f)
        X_transformed = noise_by_size_norm(freqs_qc, N_r, sigma, beta_factor)
    elif which_transform == 'log10':
        X_transformed = log10_transform(counts_qc)
    else:
        raise ValueError(f'which_transform = "{which_transform}" not implemented')

    return X_transformed


def qc_transform(counts, N_r, all_clones, sample_order, tp_dict,
                  which_transform, which_beta, qc_mask):
    '''
    Apply a clone-sum QC threshold and compute the transformed (and
    max-normalised) frequency matrix for the surviving clones.

    Shared by choose_best_clone_QC (during the QC sweep) and CloneSearch
    (once the chosen QC threshold has been applied), so the QC/transform
    logic only needs to be maintained in one place.

    Returns
    -------
    qc_clones : array of clone ids that passed QC
    X_transformed : transformed frequencies (unnormalised), indexed like qc_clones
    X_transformed_norm : X_transformed, row-wise normalised by its max
    '''

    print(qc_mask)

    freqs_all = counts / N_r
    freqs_qc = freqs_all[qc_mask, :]
    counts_qc = counts[qc_mask, :]
    qc_clones = np.array(all_clones)[qc_mask]
    print(qc_clones)

    freq_info = pd.DataFrame(freqs_qc, index=qc_clones, columns=sample_order)

    X_transformed = calculate_transformed_sequences(
        which_transform, which_beta, N_r, freq_info, freqs_qc, counts_qc,
        sample_order, tp_dict
    )
    X_transformed_norm = X_transformed - X_transformed.max(axis=1).reshape(-1, 1)

    return qc_clones, X_transformed, X_transformed_norm


def compute_qc_sweep_stats(counts, N_r, all_clones, sample_order, tp_dict,
                            which_transform, which_beta,
                            clone_sums_to_sweep=tuple(list(range(0, 15)) + [20])):
    '''
    For each candidate clone-sum QC threshold, fit a PCA to the transformed
    frequencies and compute goodness-of-fit statistics (KS, Cramer-von Mises,
    Anderson-Darling) between the radius distribution and its theoretical
    chi distribution.

    Returns a DataFrame with columns: QC, ksstats, cmstats, andersonstats.
    '''
    rows = []
    for clone_min in clone_sums_to_sweep:
        qc_mask = counts.sum(axis=1) > clone_min
        qc_clones, _, X_transformed_norm = qc_transform(
            counts, N_r, all_clones, sample_order, tp_dict,
            which_transform, which_beta, qc_mask
        )

        pca = PCA(whiten=True)
        pca_fit = pca.fit_transform(X_transformed_norm)
        n = pca_fit.shape[1]
        R = find_radius(pca_fit)

        chi_cdf = stats.chi(df=n).cdf  # build once, reuse across all 3 stats

        rows.append({
            'QC': clone_min,
            'n_clones': len(qc_clones),
            'ksstats': stats.kstest(R, chi_cdf).statistic,
            'cmstats': stats.cramervonmises(R, chi_cdf).statistic,
            'andersonstats': anderson_darling_stat(R, chi_cdf),
        })

    return pd.DataFrame(rows)


def select_best_qc(qc_stats_df, stat_col='ksstats'):
    '''
    Pick the QC threshold with the lowest goodness-of-fit statistic
    (best fit to theoretical chi distribution). Ties are broken by
    preferring the smallest (least strict) QC threshold.
    '''
    tied = qc_stats_df.loc[qc_stats_df[stat_col] == qc_stats_df[stat_col].min()]
    return tied.sort_values('QC')['QC'].values[0]


def plot_qc_sweep(qc_stats_df, chosen_QC):
    '''
    Plot KS / Cramer-von Mises / Anderson-Darling statistics across the
    QC sweep, with a marker at the chosen QC threshold.
    '''
    ax = plt.subplot()
    ax.plot(qc_stats_df['QC'], qc_stats_df['ksstats'],
            c='tab:blue', marker ='o', 
            label='Kolmogorov-Smirnov test')
    ax1 = ax.twinx()
    ax1.plot(qc_stats_df['QC'], qc_stats_df['cmstats'],
             c='tab:orange', marker ='o', 
             label='Cramér-von Mises W²-statistic')
    ax1.plot(qc_stats_df['QC'], qc_stats_df['andersonstats'],
             c='tab:purple', marker ='o', 
             label='Anderson-Darling A²-statistic')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax.axvline(chosen_QC, ls=':', c='k')
    plt.show()


def choose_best_clone_QC(counts, N_r, all_clones, sample_order, tp_dict,
                          which_transform, which_beta, make_plot=True):
    '''
    Calculate the PCA and radius fits at various clone QC thresholds and
    automatically determine which one to use.

    Set make_plot=False to skip rendering the diagnostic plot (e.g. when
    running this in a batch/headless pipeline over many samples).
    '''
    qc_stats = compute_qc_sweep_stats(
        counts, N_r, all_clones, sample_order, tp_dict,
        which_transform, which_beta
    )
    chosen_QC = select_best_qc(qc_stats)

    if make_plot:
        plot_qc_sweep(qc_stats, chosen_QC)

    return chosen_QC


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

    if pval_or_fdr.lower() == 'pvalue':
        use_FDR = False
    elif pval_or_fdr.lower() == 'fdr':
        use_FDR = True
    else:
        raise ValueError(
            f'pval_or_fdr="{pval_or_fdr}" not recognised. '
            'Choose "fdr" or "pvalue".'
        )

    R, outlier_vector, R_thresh = \
        find_gaussian_outliers(
            pca_fit,
            statistical_threshold=statistical_threshold,
            use_FDR=use_FDR
            )

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
                 which_QC = 'strict',
                 which_transform = 'g',
                 make_qc_plot=True
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
    which_QC = include small clones or not - alternatives: ["strict", "tune", "none"]. Defaults to "strict".
    which_transform = use default g(f) transform or log10 - alternatives: g(f), log10
    make_qc_plot = whether to render the QC-sweep diagnostic plot

    RETURNS:

    - list of outlier clones identified
    - PCA-transformed frequencies for each clone indicating the radius of each clone
    - radius threshold used
    - frequencies transformed by g(f)
    '''

    if which_QC == 'strict':
        # clones that are present with count >=3 at more than one timepoint
        mask = (X_counts > 2).sum(axis=1) > 1
    elif which_QC == 'tune':
        # this QC allows me to get more of the small clones
        # clones that are present at more than one timepoint
        chosen_QC = choose_best_clone_QC(
                X_counts, N_r, all_clones, sample_order, tp_dict,
                which_transform, which_beta, make_plot=make_qc_plot
            )
        mask = X_counts.sum(axis=1) > chosen_QC
    elif which_QC =='none':
        # all clones - assume user wants to run on everything
        mask = (X_counts > 0).sum(axis=1) > 0
    else:
        raise ValueError(
            'The parameter which_QC has an unrecognised value. '\
            'Please choose one of [strictQC, looseQC, noQC]'
            )

    qc_clones, X_transformed, X_transformed_norm = qc_transform(
        X_counts, N_r, all_clones, sample_order, tp_dict,
        which_transform, which_beta, mask
    )

    pca_fit, R_thresh, outlier_vector = pca_outlier_identification(
        X_transformed_norm, qc_clones, pval_or_fdr, statistical_threshold
    )
    outlier_list = qc_clones[outlier_vector]

    X_transformed = pd.DataFrame(X_transformed, index=qc_clones, columns=sample_order)

    return outlier_list, pca_fit, R_thresh, X_transformed


def _stable_cluster_labels(fl, row_linkage):
    '''
    Relabel cluster ids so that cluster numbering follows the order
    clusters first appear in the dendrogram leaf order, giving stable,
    reproducible cluster numbers regardless of fcluster's internal ids.
    '''
    dendrogram_order = leaves_list(row_linkage)
    remap = {}
    counter = 1
    for idx in dendrogram_order:
        cluster_id = fl[idx]
        if cluster_id not in remap:
            remap[cluster_id] = counter
            counter += 1
    return np.array([remap[c] for c in fl])

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
    fl = _stable_cluster_labels(fl, row_linkage_all)

    return row_linkage_all, fl
