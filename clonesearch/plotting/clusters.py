'''
Plotting functions to show heatmap of clustered TCRs by trajectory.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import leaves_list
from matplotlib import cm, colors

def plot_clustermap(input_df, row_linkage, row_annotations:pd.DataFrame, annotation_colors:dict,
                    cmap = 'Greys', cbar_title = r'$g(f) - g(f_{max})$',
                    show_clusters = True, cluster_col = 'cluster', cluster_colors = None,
                    show_radius = False,
                    radius_col = 'radius',
                    radius_kwargs = {'cmap':'Blues', 'vmin' : 0, 'vmax' : 10},
                    ):

    '''
    input_df contains the frequencies of sequences to show in clustermap. 
        The colorbar assumes these are g(f)-g(f_max). 
        If not using transformed frequencies, the colormap title can be changed with cbar_title.
    row_linkage contains the clustering of TCRs. Can be obtained from CloneSearch_clustering.
    row_annotations contains information about the clones. 
        row_annotation indexes must be the same as input_df indexes
    annotation_colors contains the colors to use for row_annotations.   
        The keys of the dict need to correspond to row_annotation columns
    To add a legend on which cluster colors for the clustering, 
        please provide cluster_colors as a mapping between cluster names and cluster colors
    '''

    row_annotations = row_annotations.copy()
    for c in row_annotations.columns:
        if (c in annotation_colors) & (c not in [cluster_col, radius_col]):
            row_annotations[c] = row_annotations[c].map(annotation_colors[c])

    if show_radius:
        def generate_colors(vals, cmap, vmin, vmax):
            cmap = cm.get_cmap(cmap)
            cnorm = colors.Normalize(vmin=vmin, vmax=vmax)
            vals = np.array(vals, dtype=float)
            normed = cnorm(vals)
            rgb_colors = cmap(normed)[:, :3]
            return rgb_colors, cmap, cnorm

        radii = row_annotations[radius_col].values
        rgb_colors, cmap_radius, cnorm_radius = generate_colors(radii, radius_kwargs['cmap'], radius_kwargs['vmin'], radius_kwargs['vmax'])
        row_annotations[radius_col] = [tuple(r) for r in rgb_colors]

    if show_clusters:
        if cluster_colors is None:
            unique_clusters = sorted(row_annotations[cluster_col].unique(), key=int)
            num_clusters = len(unique_clusters)
            all_colors = [c for cmap in [plt.cm.tab20b, plt.cm.tab20c, plt.cm.tab20]
                          for c in [cmap(i) for i in range(20)]]
            cluster_colors = dict(zip(unique_clusters, all_colors[:num_clusters]))

        original_cluster_labels = row_annotations[cluster_col].values.copy()
        row_annotations[cluster_col] = [cluster_colors[x] for x in row_annotations[cluster_col]]

    g = sns.clustermap(input_df, col_cluster=False, row_linkage=row_linkage,
                       row_colors=row_annotations, square=False,
                       colors_ratio=0.04,
                       dendrogram_ratio=(.1, .15),
                       figsize=(8, 8),
                       yticklabels=False,
                       cbar_kws={"orientation": "horizontal"}, cmap=cmap)

    # Axis position references
    heatmap_pos = g.ax_heatmap.get_position()
    dendro_pos = g.ax_row_dendrogram.get_position()

    cbar_y = 0.92
    cbar_height = 0.02
    tick_fontsize = 11
    tick_length = 5

    # g(f) colorbar — over heatmap
    g.ax_cbar.set_position([heatmap_pos.x0, cbar_y, heatmap_pos.width, cbar_height])
    g.ax_cbar.set_title(cbar_title, fontsize=14)
    g.ax_cbar.tick_params(axis='x', length=tick_length, labelsize=tick_fontsize)

    g.ax_heatmap.set_xticklabels(input_df.columns, rotation=90, fontsize=16)
    g.ax_row_colors.tick_params(axis='x', labelsize=14)

    if show_clusters:

        display_order = leaves_list(row_linkage)

        cluster_positions = {}
        for display_pos, original_idx in enumerate(display_order):
            cluster = original_cluster_labels[original_idx]
            if cluster not in cluster_positions:
                cluster_positions[cluster] = []
            cluster_positions[cluster].append(display_pos)

        col_names = list(row_annotations.columns)
        cluster_col_idx = col_names.index(cluster_col)
        n_cols = len(col_names)

        x_pos = (cluster_col_idx + 0.5) / n_cols

        n_rows = len(display_order)
        for cluster, positions in cluster_positions.items():
            if len(positions) >= 2:
                mid = np.mean(positions) / n_rows
                g.ax_row_colors.text(
                    x_pos, 1 - mid - 0.005,
                    str(cluster),
                    transform=g.ax_row_colors.transAxes,
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='white'
                )

    if show_radius:
        row_colors_pos = g.ax_row_colors.get_position()

        fig_margin = 0.01
        dendro_x0 = max(dendro_pos.x0, fig_margin)
        combined_x1 = row_colors_pos.x0 + row_colors_pos.width
        safe_width = combined_x1 - dendro_x0

        cbar_ax = g.fig.add_axes([
            dendro_x0,
            cbar_y,
            safe_width,
            cbar_height
        ])
        cbar = g.fig.colorbar(
            plt.cm.ScalarMappable(norm=cnorm_radius, cmap=cmap_radius),
            cax=cbar_ax,
            orientation='horizontal',
        )
        cbar.set_label(radius_col, fontsize=14)
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.tick_params(axis='x', length=tick_length, labelsize=tick_fontsize)
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

    return g
