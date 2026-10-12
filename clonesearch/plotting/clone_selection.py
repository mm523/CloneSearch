import matplotlib.pyplot as plt

def plot_qc_sweep(qc_stats_df, chosen_QC, plot_additional_stats = False, save_path = None):
    '''
    Plot KS / Cramer-von Mises / Anderson-Darling statistics across the
    QC sweep, with a marker at the chosen QC threshold.
    '''
    ax = plt.subplot()
    ax.plot(qc_stats_df['QC'], qc_stats_df['ksstats'],
            c='tab:blue', marker ='o', 
            label='Kolmogorov-Smirnov D-statistic')
    if plot_additional_stats:
        ax1 = ax.twinx()
        ax1.plot(qc_stats_df['QC'], qc_stats_df['cmstats'],
                c='tab:orange', marker ='o', 
                label='Cramér-von Mises W²-statistic')
        ax1.plot(qc_stats_df['QC'], qc_stats_df['andersonstats'],
                c='tab:purple', marker ='o', 
                label='Anderson-Darling A²-statistic')

    lines, labels = ax.get_legend_handles_labels()
    if plot_additional_stats:
        lines2, labels2 = ax1.get_legend_handles_labels()
        lines = lines + lines2
        labels = labels + labels2
    ax.legend(lines, labels, loc='best')
    ax.axvline(chosen_QC, ls=':', c='k')
    ax.set_xlabel('Minimum total clone count')

    if save_path is not None:
        plt.savefig(f'{save_path}/cloneQC_threshold_sweep.pdf', bbox_inches = 'tight')
    plt.show()