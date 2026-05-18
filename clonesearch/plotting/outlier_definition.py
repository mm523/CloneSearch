import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from clonesearch.utils.gaussian_outliers import P_r

def plot_R_and_fit(R, R_thresh, 
                   n_dims, 
                   stat_thresh = 0.05, 
                   show = False):
    theoretical_curve = P_r(sorted(R), n = n_dims)
    num_outliers = sum(R > R_thresh)

    gkde = gaussian_kde(R)
    kde = gkde.pdf(sorted(R))
    plt.plot(sorted(R), kde, c = 'k', label = 'R distribution')
    plt.plot(sorted(R), theoretical_curve, c = 'r', label = 'theoretical curve')
    plt.axvline(R_thresh, c = 'r', ls = ':', label = f'radius threshold - FDR = {stat_thresh}')
    plt.legend()

    txt = 'threshold = ' + str(R_thresh.round(2)) + \
            '\nNumber of outliers = ' + str(num_outliers)
    plt.text(R_thresh+.5, theoretical_curve.max(), txt, va = 'top')
    plt.legend(bbox_to_anchor = [0.5, -.1], loc = 'upper center', ncols =2)
    if show:
        plt.show()

def plot_pca(pca_fit, explained_variance, 
             outlier_vector, radius, 
             clone_annotations = None, colors = None,
             num_PCs=None, show_density=True, 
             plot_radius = False, highlight_outliers = True, 
             title = None, show = False):

    if num_PCs == None:
        num_PCs = pca_fit.shape[1]

    axis_pairs = list(zip([x for x in range(num_PCs) if x%2 == 0], [x for x in range(num_PCs) if x%2 != 0]))
    lim = max(np.abs(pca_fit.min()), np.abs(pca_fit.max())) + 1

    nrows = round(len(axis_pairs)/3) if len(axis_pairs) % 3 == 0 else round((len(axis_pairs) + 2)/3)
    ncols = 3
    if len(axis_pairs) < 3:
        ncols = len(axis_pairs)

    f, ax = plt.subplots(ncols=ncols, nrows=nrows, 
                         figsize = (5*ncols, 4.5*nrows + 1), 
                         sharex = True, sharey = True)
    if len(axis_pairs) == 1:
        axs = ax
    else:
        axs = ax.ravel()

    for i in range(len(axis_pairs)):
        if len(axis_pairs) > 1:
            myax = axs[i]
        else:
            myax = axs
        x,y = axis_pairs[i]
        myax.scatter(pca_fit[:, x], pca_fit[:, y], 
                    s = 2, alpha = 0.1, ec = 'grey', label = 'all points')
        if highlight_outliers:
            myax.scatter(pca_fit[outlier_vector, x], pca_fit[outlier_vector, y], 
                        s = 8, ec = 'red', label = 'outliers')
        axs0 = myax

        if show_density == True:
            axs1 = myax.twinx()
            axs2 = myax.twiny()
            sns.kdeplot(pca_fit[:, x], ax = axs1, color='grey', fill='grey')
            sns.kdeplot(y = pca_fit[:, y], ax = axs2, color='grey', fill='grey')
            axs1.set_ylim(0,2)
            axs2.set_xlim(0,2)

            if clone_annotations:
                for k in clone_annotations.keys():
                    sns.kdeplot(pca_fit[clone_annotations[k], x], ax = axs1, color=colors[k], label = k)
                
                for k in clone_annotations.keys():
                    if k in colors.keys():
                        sns.kdeplot(y = pca_fit[clone_annotations[k], y], ax = axs2, color=colors[k], label = k)

        if plot_radius:
            limit = plt.Circle((0,0), radius, ec = 'red', fill=False)
            myax.add_patch(limit)

        myax.set_xlim((-lim, lim))
        myax.set_ylim((-lim, lim))
        myax.set_xlabel('PC' + str(x+1) + ' ({}%)'.format((explained_variance[x]*100).round(2)), fontsize=14)
        myax.set_ylabel('PC' + str(y+1) + ' ({}%)'.format((explained_variance[y]*100).round(2)), fontsize = 14)

    handles1, l1 = axs0.get_legend_handles_labels()
    if show_density == True:
        handles2, l2 = axs2.get_legend_handles_labels()
    else:
        handles2, l2 = [], []

    handles = handles1 + handles2
    l = l1 + l2

    leg = f.legend(handles, l, bbox_to_anchor = [0.5,0], loc='upper center', ncols = 5, fontsize = 14)
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
    if title:
        f.suptitle(title, fontsize = 20)
    plt.tight_layout()
    if show:
        plt.show()
