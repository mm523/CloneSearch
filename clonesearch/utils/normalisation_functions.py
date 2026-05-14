'''
g(f) and log-transform functions used to pre-process frequency data before 
passing to PCA
'''

import numpy as np

def g_of_f(f, N_r, s, beta_factor):
    r'''
    Integral of the noise based on the frequencies. 

    .. math::
        g(f)= \frac{2}{\sigma} \ln\left(\frac{\sqrt{\frac{\beta}{N_{r}} + \sigma^2}-\sigma}{\sqrt{\frac{\beta}{N_{r}} + f \sigma^2}-\sigma\sqrt{f}} \right)
    '''

    prefactor = -2/s
    main = -np.sqrt(f)*s + np.sqrt(beta_factor/N_r + f*s**2)

    g_of_1 = prefactor*np.log(-np.sqrt(1)*s + np.sqrt(beta_factor/N_r + 1*s**2))

    return prefactor * np.log(main) - g_of_1

def log10_transform(X):
    '''
    Standard log10 transformation of frequencies. 
    It uses a pseudocount of 1/3 for counts = 0.
    The N_r for the frequency calculation is calculated by summing over the pseudocounts too.
    '''

    X = X.astype(float)
    X[X == 0] = 1/3
    X_f = X/X.sum(axis=0)

    f_log10 = np.log10(X_f)
    return f_log10

def noise_by_size_norm(df, N_r, s, beta_factor):
    '''
    df = frequency array
    N_r = total counts per sequencing sample
    s = sigma
    '''

    new_df = []
    for i in range(df.shape[1]):
        fs = df[:,i]
        nr = N_r[i]
        bf = beta_factor[i]

        fs_norm = g_of_f(fs, nr, s, bf)
        new_df.append(fs_norm)

    X = np.array(new_df).T
    assert X.shape == df.shape
    return X
