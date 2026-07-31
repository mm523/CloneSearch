'''
This module contains the functions that, assuming a multivariate gaussian distribution, 
calculate which TCRs are outliers in the PCA space
'''

import numpy as np
from scipy.special import gammainccinv,gammainc
from scipy.special import gamma as gamma_fn
from scipy.stats import ecdf

def P_r(r, n):
    ''' 
    Calculates the expected radius distribution P(r).

    n = number of dims
    r = radii -- an array
    '''
    s = 1 # sigma of the radius = 1 since PCA is whitened
    r = np.array(r)
    S_n_minus_1 = (2*np.pi**(n/2))/gamma_fn(n/2)
    denom = (2*np.pi*s**2)**(n/2)
    num = r**(n-1)*np.exp((-r**2)/(2*s**2))
    Pr = 1/denom * S_n_minus_1 * num

    return Pr

def find_radius(pca_df):
    '''
    Turn the PCA vectors into polar coordinates and get radius r for each point
    '''

    r = np.linalg.norm(pca_df.values if hasattr(pca_df, 'values') else pca_df, axis=1)
    assert len(r) == pca_df.shape[0]
    return r

def find_radius_outlier_thresh_pval(dimensions, pval = 0.025):
    '''
    Find the outlier points using a pvalue threshold
    '''

    sd = 1 # sigma of the radius = 1 since PCA is whitened
    x = gammainccinv(dimensions/2, pval)
    r_thresh = sd*np.sqrt(2*x)

    return r_thresh

def find_fdr_thresh(radii, dims, fdr):
    '''
    Find the outlier points using a BH-based FDR threshold. 
    Returns an analytical threshold from the theoretical distribution — 
    not one of the observed radii — so the > vs >= boundary ambiguity in 
    find_gaussian_outliers does not arise.
    '''
    n = len(radii)
    p_values = 1 - theoretical_cdf(radii, dims)
    sorted_p = np.sort(p_values)                        # ascending
    bh_line  = np.arange(1, n + 1) / n * fdr
    below    = sorted_p <= bh_line
    if not np.any(below):
        return float(np.max(radii))                     # no rejections: flag nothing
    k = int(np.where(below)[0][-1]) + 1                 # largest k (1-indexed)
    return float(find_radius_outlier_thresh_pval(dims, k * fdr / n))

def find_gaussian_outliers(pca_fit, statistical_threshold = 0.05, use_FDR = True):
    '''
    Find the outlier sequences given a whitened and centered PCA
    '''

    dims = pca_fit.shape[1]

    radii = find_radius(pca_fit)
    if not use_FDR:
        radius_thresh = find_radius_outlier_thresh_pval(dims, statistical_threshold)
    elif use_FDR:
        radius_thresh = find_fdr_thresh(radii, dims, statistical_threshold)

    outlier_vector = radii > radius_thresh

    return radii, outlier_vector, radius_thresh

def theoretical_cdf(w,n):
    '''
    Calculates the integral over P(r), which is what I use to find the threshold

    n = number of dimensions (i.e. number of samples)
    w = values of radius
    '''

    w = np.array(w)

    s = 1 # sigma of the radius = 1 since PCA is whitened
    complete = gammainc(n/2, (w**2)/(2*(s**2)))

    return complete
