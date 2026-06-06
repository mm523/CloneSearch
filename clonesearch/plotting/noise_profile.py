import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from clonesearch.utils.noise_functions import noise_func

def plot_noise_profile(avgs_dt_freqs, 
                       max_dt = None, min_dt = None, 
                       title = None,
                       ax = None,
                       xlims = (-5,-1), show = False):

    if not ax:
        f, ax = plt.subplots(figsize = (8,6))

    x = np.logspace(xlims[0], xlims[1])

    if min_dt:
        avgs_dt_freqs = avgs_dt_freqs.loc[avgs_dt_freqs['dt'] > min_dt]
    if max_dt:
        avgs_dt_freqs = avgs_dt_freqs.loc[avgs_dt_freqs['dt'] < max_dt]

    X = avgs_dt_freqs['freqs']
    Y = avgs_dt_freqs['avgs']
    dt = avgs_dt_freqs['dt']

    fit_freqs = X.copy()
    fit_avgs = Y.copy()
    fit_freqs = fit_freqs[fit_avgs > 0]
    fit_avgs = fit_avgs[fit_avgs > 0]

    myfit = curve_fit(noise_func, fit_freqs, np.log(fit_avgs), bounds = [(0,0),(1,np.inf)])
    ss, b = myfit[0]

    y1 = (ss*x)**2
    y = b*x

    ax.plot(x,y, ls = ':', c = 'k', lw = 1, 
                label = f'Poisson noise, $b = {b:.2g}$')
    ax.plot(x,y1, ls = '--', c = 'k', lw = 1, 
                label = f'Multiplicative noise, $\sigma = {ss:.2g}$')

    sc = ax.scatter(X, Y, c = dt, s = 30, cmap = 'Reds', ec = 'k', lw = .5)

    ax.set_xlabel('mean f at t', fontsize=14)
    ax.set_ylabel(r'$\delta f^2$', fontsize=14)

    ax.set_yscale('log')
    ax.set_xscale('log')

    y4 = b*x + ss**2*x**2
    ax.plot(x,y4, c = 'k', lw = 2,
                label = f'Mixed noise, $b = {b:.2g}$, $\sigma = {ss:.2g}$')

    ax.legend(fontsize=12)
    if title:
        ax.set_title(title)
    cbar=plt.colorbar(sc, label = 'Time difference between timepoints (days)')
    cbar.set_label('Time difference between timepoints (days)', size=12)
    cbar.ax.tick_params(labelsize=12) 

    if not ax:
        plt.tight_layout()
    if show:
        plt.show()

    return ss, b
