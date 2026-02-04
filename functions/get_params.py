import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from functions.noise_functions import calc_diffs, make_df, noise_func, var_btn_tps

def get_sigma_and_beta(freq_info, tp_dict, perc_step = 10, min_perc = 0):

    sample_order = freq_info.columns

    avgs, dt, f, percentiles, N = calc_diffs(var_btn_tps, freq_info, 
                                             sample_order, tp_dict, 
                                            use_percentiles=True, perc_step=perc_step, min_perc = min_perc)
    
    N_df = make_df(N, 'N')
    avgs_df = make_df(avgs, 'avgs')
    dt_df = make_df(dt, 'dt')
    freqs_df = make_df(f, 'freqs')

    avgs_df = pd.merge(avgs_df, N_df)
    avgs_dt = pd.merge(avgs_df, dt_df)
    avgs_dt_freqs = pd.merge(avgs_dt, freqs_df)
    avgs_dt_freqs['percentiles'] = avgs_dt_freqs['p1'] + '-' + avgs_dt_freqs['p2']

    X = avgs_dt_freqs['freqs']
    Y = avgs_dt_freqs['avgs']

    fit_freqs = X.copy()
    fit_avgs = Y.copy()

    fit_freqs = fit_freqs[fit_avgs > 0]
    fit_avgs = fit_avgs[fit_avgs > 0]
    
    myfit = curve_fit(noise_func, fit_freqs, np.log(fit_avgs), bounds = [(0,0),(1,np.inf)])
    fit_s, fit_b = myfit[0]
    
    return fit_s, fit_b