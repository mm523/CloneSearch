'''
The following script takes a list of sequences and information on their trajectories, 
clusters them and then outputs the cluster lists and a plot of the clustered information.

The following arguments are user specified and are needed: 
- -i/--input: Path to file to use for calculating distances. 
              It can be any .csv file of frequencies, transformed frequencies or PCA projections.

The following arguments can be toggled by the user:
- --clone-list: List of clones to cluster. 
                If "all", all rows of the input will be clustered. Defaults to "all". 
                If provided, row names of input_df must correspond to clone names.
- --normalise: Whether to normalise each row in the input by the max. 
               Defaults to True. Recommended if using transformed frequencies.
- --title: The title of the colorbar. Default is "$g(f) - g(f_{\text{max}})$". 
           It can be toggled by the user. Only used for plotting purposes.
- --distance-metric: Distance metric to use for clustering. 
                     Passed to scipy.spatial.distance.pdist. 
                     For transformed and normalised frequencies, we recommend "correlation". Defaults to "correlation".
- --linkage-method: Linkage method to use for clustering. 
                    Passed to scipy.cluster.hierarchy.linkage. 
                    For transformed and normalised frequencies, we recommend "average". Defaults to "average".
- --distance-threshold: Distance threshold to use to define clusters. 
                        Passed to scipy.cluster.hierarchy.fcluster. Defaults to 0.61.
- --output-folder: Path to save results. Defaults to current folder.
'''

from optparse import OptionParser
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from clonesearch.CloneSearch import CloneSearch_clustering
from clonesearch.plotting.clusters import plot_clustermap

def parse_all_arguments():
    parser = OptionParser(conflict_handler='resolve')

    parser.add_option('-i', '--input',
                      dest='input',
                      help='Path to file to use for calculating distances')
    parser.add_option('--normalise', default='True',
                      choices=['True', 'False'],
                      dest='norm',
                      help='Whether to normalise each row in the input by the max. ' \
                      'Defaults to True. ' \
                      'Recommended if using transformed frequencies.')
    parser.add_option('--title', default=r'$g(f) - g(f_{\text{max}})$',
                      dest='title',
                      help='Whether to normalise by the max the input file. ' \
                      'Defaults to True. ' \
                      'Recommended if using transformed frequencies.')
    parser.add_option('--clone-list', default = 'all',
                      dest='clone_list',
                      help='List of clones to cluster. ' \
                      'If "all", all rows of the input will be clustered. ' \
                      'Defaults to "all". ' \
                      'If provided, row names of input_df must correspond to clone names.')
    parser.add_option('--distance-metric',
                      default='correlation',
                      dest = 'distance',
                      help = 'Distance metric to use for clustering. ' \
                      'Passed to scipy.spatial.distance.pdist. ' \
                      'For transformed and normalised frequencies, we recommend "correlation". ' \
                      'Defaults to "correlation".'
                      )
    parser.add_option('--linkage-method',
                      default='average',
                      dest = 'linkage',
                      help = 'Linkage method to use for clustering. ' \
                      'Passed to scipy.cluster.hierarchy.linkage. ' \
                      'For transformed and normalised frequencies, we recommend "average". ' \
                      'Defaults to "average".'
                      )
    parser.add_option('--distance-threshold',
                      default=0.61,
                      dest = 'distance_threshold',
                      help = 'Distance threshold to use to define clusters. ' \
                      'Passed to scipy.cluster.hierarchy.fcluster. ' \
                      'Defaults to 0.61.'
                      )
    parser.add_option('-o', '--output-folder', default = '.',
                      dest='output_path', help='Path to save results. Defaults to current folder.')

    (options, args) = parser.parse_args()

    return vars(options)


def main():

    args_dict = parse_all_arguments()
    output_path = Path(args_dict['output_path'])

    complete_df = pd.read_csv(args_dict['input'], index_col=0)
    if args_dict['clone_list'] != 'all':
        clone_list = pd.read_csv(args_dict['clone_list'],
                                 sep = '\t', header = None)[0].tolist()
        input_df = complete_df.loc[clone_list]
    else:
        input_df = complete_df.copy()

    if args_dict['norm'] == 'True':
        index = input_df.index
        columns = input_df.columns
        X = input_df.values
        X_normed = X - X.max(axis=1).reshape(-1, 1)
        input_df = pd.DataFrame(X_normed,
                                index = index, columns=columns)

    distance_metric = args_dict['distance']
    linkage_method = args_dict['linkage']
    distance_threshold = args_dict['distance_threshold']

    row_linkage_all, fl = CloneSearch_clustering(input_df,
                                        distance_metric = distance_metric,
                                        linkage_method = linkage_method,
                                        distance_thresh = distance_threshold)

    row_annotations = pd.DataFrame(fl, index = input_df.index, columns=['cluster'])
    clone_colors = {}

    plot_clustermap(input_df, row_linkage_all, row_annotations,
                    clone_colors, cluster_col='cluster', cbar_title=args_dict['title'])
    plt.savefig(output_path/'heatmap_outliers_and_previously_ann.pdf', bbox_inches = 'tight')
    plt.close()

    input_df['cluster'] = fl
    input_df.to_csv(output_path/'clustered_clones.csv')

if __name__ == '__main__':
    main()
