import numpy as np
from itertools import combinations
import pandas as pd

### oscillation functions
def var_btn_tps(x,y):
    return (y-x)**2

### functions for percentiles
def _get_vals_for_perc(df_freq : pd.DataFrame, tp : list):
    mask = np.any(df_freq[tp].values > 0, axis=1)
    if not np.any(mask):
        return np.array([]), mask, np.array([])
    vals = df_freq.loc[mask, tp].values
    means = vals.mean(axis=1)
    return vals, mask, means

def _get_percentile_values(means : np.array, perc_step : float, min_perc = 0):
    myp = np.unique(np.clip(np.arange(min_perc, 100 + perc_step, perc_step), 0, 100))
    myp_vals = np.percentile(np.unique(means), myp)
    percentiles = list(zip(map(str, myp[1:]), map(str, myp[:-1])))
    percentile_vals = list(zip(myp_vals[1:], myp_vals[:-1]))

    return percentiles, percentile_vals

def _get_sets(df_freq:pd.DataFrame, tp:list, percentiles:list[tuple], percentile_vals:list[tuple], mask:np.array):
    means_all = np.mean(df_freq[tp].values, axis=1)

    # Extract lower and upper percentiles as arrays for faster broadcasting
    p0_vals = np.array([p0 for _, p0 in percentile_vals])
    p1_vals = np.array([p1 for p1, _ in percentile_vals])

    # Vectorized conditions using broadcasting
    condition1 = (means_all[:, None] <= p1_vals) & (means_all[:, None] > p0_vals)  # Handles condition for non-zero percentiles
    condition2 = (means_all[:, None] <= p1_vals) & (means_all[:, None] >= p0_vals)  # Handles condition for zero percentiles

    # Apply conditions based on percentiles
    sets = np.where(np.array([float(percentiles[i][1]) == 0 for i in range(len(percentiles))]), condition2, condition1)

    # Apply the mask directly
    sets = sets & mask[:, None]
    sets = [sets[:,i] for i in range(sets.shape[1])]

    return sets

def find_percentiles(df_freq, tp, perc_step, min_perc):
    vals, mask, means = _get_vals_for_perc(df_freq, tp)
    assert means.shape[0] == len(vals)

    percentiles, percentile_vals = _get_percentile_values(means, perc_step, min_perc)
    percentiles_ss = np.array(percentiles)
    percentile_vals = np.array(percentile_vals)
    perc_mask0 = np.not_equal(percentile_vals[:, 0], percentile_vals[:, 1])
    if min_perc == 0:
        perc_mask1 = percentiles_ss[:, 1].astype(float) == 0
        perc_mask = perc_mask1+perc_mask0
    else:
        perc_mask = perc_mask0
    percentiles_ss = [tuple(x) for x in percentiles_ss[perc_mask]]
    percentile_vals = [tuple(x) for x in percentile_vals[perc_mask]]
    sets = _get_sets(df_freq, tp, percentiles_ss, percentile_vals, mask)
    
    if min_perc == 0:
        assert sum(mask) == np.array(sets).sum()
    assert np.array([len(s) == df_freq.shape[0] for s in sets]).all()
    # return the subset percentiles with mask for further operations
    # and also return all the percentiles so that I have record of eval percentiles
    # across all the timepoint pairs
    return sets, percentiles_ss, percentiles

### functions to get noise oscillations
def _output(func, x, y):
    mask = x+y > 0
    x = x[mask]
    y = y[mask]
    return func(x,y).mean()
def _freqs(vals):
    mask = vals.sum(axis=1) > 0
    meanf = vals[mask].mean(axis=1)
    return meanf.mean()
def _dt(t1, t2, tp_dict):
    return np.abs(tp_dict[t1] - tp_dict[t2])

def _no_percentile_output(func, mydf, tp_pairs, tp_dict, dt_min):

    outputs = np.array([_output(func, mydf[x], mydf[y]) for x,y in tp_pairs])
    dt = np.array([_dt(x,y,tp_dict) for x,y in tp_pairs])
    freqs = np.array([_freqs(mydf[[x,y]]) for x,y in tp_pairs])
    assert len(freqs) == len(outputs) == len(dt)

    mask = dt >= dt_min

    return outputs[mask], dt[mask], freqs[mask]

def _get_single_perc_output(func, mydf, t1, t2, ss):
    t1_vals = mydf[t1].values[ss]
    t2_vals = mydf[t2].values[ss]
    t1_t2_vals = mydf[[t1,t2]].values[ss,:]
    assert t1_t2_vals.shape[0] == sum(ss)

    o = _output(func, t1_vals, t2_vals)
    f = _freqs(t1_t2_vals)

    return o, f

def _get_percentiles_output(func, sets, percentiles, min_perc, t1, t2, tp_dict, mydf):
    dt_value = _dt(t1, t2, tp_dict)

    dt_dict = {}
    N_dict = {}
    o_dict = {}
    f_dict = {}

    for ss, p in zip(sets, percentiles):
        ss_sum = sum(ss)
        if ss_sum == 0 or float(p[1]) < min_perc: continue
        # My bin edges are not unique i.e. certain quantiles are identical, so I have empty bins 
        # remove perc_min to reduce comp time

        dt_dict[p] = dt_value
        N_dict[p] = ss_sum

        o, f = _get_single_perc_output(func, mydf, t1, t2, ss)
        o_dict[p] = o
        f_dict[p] = f
    
    return dt_dict, N_dict, o_dict, f_dict


def calc_diffs(func, mydf, tp_names, tp_dict, use_percentiles = False, perc_step = None, min_perc = 0, dt_min=0):
    
    tp_pairs = [(t1, t2) for t1, t2 in combinations(tp_names, 2) if np.abs(tp_dict[t1] - tp_dict[t2]) >= dt_min]

    if use_percentiles == False:
        return _no_percentile_output(func, mydf, tp_pairs, tp_dict, dt_min)
    
    else:
        outputs = {pair: None for pair in tp_pairs}
        dt = {pair: None for pair in tp_pairs}
        freqs = {pair: None for pair in tp_pairs}
        N = {pair: None for pair in tp_pairs}

        for t1,t2 in tp_pairs:
            
            sets, percentiles_ss, percentiles_all = find_percentiles(mydf, [t1,t2], perc_step, min_perc)

            result = _get_percentiles_output(func, sets, percentiles_ss, min_perc, t1, t2, tp_dict, mydf)
            
            dt[t1, t2], N[t1, t2], outputs[t1, t2], freqs[t1, t2] = result

        return outputs, dt, freqs, percentiles_all, N
    

### other functionalities
def make_df(mydict, name):
    X = pd.DataFrame(mydict).reset_index(names=['p1','p2'])
    X = X.melt(id_vars=[('p1',''), ('p2','')])
    X.columns = ['p1','p2','t1','t2', name]
    # NA arise from making the df
    X = X.dropna(subset = name)
    return X

def noise_func(f, s, b):
    return np.log(b*f + s**2*f**2)