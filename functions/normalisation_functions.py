import numpy as np

def noise_by_size_norm(df, N_r, s, beta_factor):
    '''
    df = frequency array
    N_r = total counts per sequencing sample
    s = sigma
    '''
    def _norm(f, N_r, s, beta_factor):
        # this is the integral of the noise based on the frequencies
        prefactor = -2/s
        u = f * (s**2*N_r) / beta_factor
        main = -np.sqrt(u) + np.sqrt(1+u)

        return prefactor * np.log(main)

    new_df = []
    for i in range(df.shape[1]):
        fs = df[:,i]
        nr = N_r[i]
        bf = beta_factor[i]

        fs_norm = _norm(fs, nr, s, bf)
        new_df.append(fs_norm)

    X = np.array(new_df).T
    assert X.shape == df.shape
    return X