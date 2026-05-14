'''
This module contains the functions that, assuming a multivariate gaussian distribution, 
calculate which TCRs are outliers in the PCA space
'''

import numpy as np
from scipy.special import gammainccinv,gammainc
from scipy.special import gamma as gamma_fn
from scipy.stats import ecdf

def P_r(r, s, n):
    ''' 
    Calculates a the expected radius distribution P(r).

    s = sigma
    n = number of dims
    r = radii -- an array
    '''

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

    r = np.array([np.linalg.norm(pca_df[x]) for x in range(pca_df.shape[0])])
    assert len(r) == pca_df.shape[0]
    return r

def find_radius_outlier_thresh_pval(sd, dimensions, pval = 0.025):
    '''
    Find the outlier points using a pvalue threshold
    '''

    x = gammainccinv(dimensions/2, pval)
    r_thresh = sd*np.sqrt(2*x)

    return r_thresh

def find_fdr_thresh(radii, dims, radii_sigma, fdr):
    '''
    Find the outlier points using an FDR threshold
    '''

    sorted_r = sorted(set(radii))
    theory = theoretical_cdf(sorted_r, dims, radii_sigma)
    ecdf_real = ecdf(radii)
    try:
        idx = np.where((1-theory)/(1-ecdf_real.cdf.probabilities) < fdr)[0][0]
    except:
        idx = -1
    radius_thresh = sorted_r[idx]

    return radius_thresh

def find_gaussian_outliers(pca_fit, statistical_threshold = 0.05, use_FDR = True):
    '''
    Find the outlier sequences given a whitened and centered PCA
    '''

    dims = pca_fit.shape[1]

    radii = find_radius(pca_fit)
    radii_sigma = 1
    if not use_FDR:
        radius_thresh = find_radius_outlier_thresh_pval(radii_sigma, dims, statistical_threshold)
    elif use_FDR:
        radius_thresh = find_fdr_thresh(radii, dims, radii_sigma, statistical_threshold)

    outlier_vector = radii > radius_thresh

    return radii, outlier_vector, radius_thresh

def theoretical_cdf(w,n,s):
    '''
    Calculates the integral over P(r), which is what I use to find the threshold

    n = number of dimensions (i.e. number of samples)
    s = sigma
    w = values of radius

    '''

    w = np.array(w)

    complete = gammainc(n/2, (w**2)/(2*(s**2)))

    return complete
