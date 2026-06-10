'''
The following script loads TCR Vb samples as specified in a metadata file using `load_data.py`.
It then processes the samples to find outlier TCR trajectories. 
It outputs 3 files: 
1. a .csv file with transformed sequences for each TCR in each timepoint,
2. a .csv file with the PCA projections for each TCR,
3. a list of TCR outliers, given the identifiers provided.


The following arguments are user specified and are needed: 
- --metadata: path to metadata file
- --cdr3aa: Column name for CDR3aa sequence information in TCRVb sequencing file. 
            Used for filtering of productive sequences.
- --counts: Column name containing count information

The following arguments can be 'None' and will determine how a TCR clonotype is defined:
- -v: Column name containing V gene information. 
                 If not provided, it will not be used to define a TCR clone.
- -j: Column name containing J gene information. 
                 If not provided, it will not be used to define a TCR clone.
- --cdr3: Column name containing CDR3 gene information. 
          If not provided, it will not be used to define a TCR clone. 
          The model is agnostic to this being nucleotide or aa sequences.

The following arguments can be toggled by the user:
- -d/--delimiter: Delimiter used in TCR Vb sequencing files. 
                  Defaults to tab. Options: ["tab", "comma"]
- -i/--input-folder: Path to TCR Vb sequencing files. Defaults to current folder.
- -o/--output-folder: Path to TCR Vb sequencing files. Defaults to current folder.
- --stat-thresh: p-value or FDR threshold to use for outlier selection.
- --pval-or-fdr: Whether to use p-value or FDR selection. 
                 Defaults to FDR. Options: ["fdr", "pvalue"].
- --beta-param-constant: Whether to keep the beta parameter constant across samples. 
                         Defaults to True. We recommend leaving this to True.
- --which-QC: How to select which clones are included in the analysis. 
              Options are ["strictQC", "looseQC", "noQC"]. 
              With "strictQC", clones are kept if present at count > 2 in > 1 timepoint.
              With "looseQC", clones are kept if present at count > 0 in > 1 timepoint.
              With "noQC", all clones are included. 
              This makes the implementation very slow and may give noisier results.
              Defaults to "strictQC".
- --transform: Which transformation to apply to the frequencies. 
               Options are ["g", "log"]. Defaults to "g", which implements g(f).
'''

from optparse import OptionParser
from pathlib import Path
import pandas as pd
from clonesearch.load_data import load_data
from clonesearch.CloneSearch import CloneSearch

def parse_all_arguments():
    parser = OptionParser(conflict_handler='resolve')

    parser.add_option('--metadata', dest='metadata', help='path to metadata file')
    parser.add_option('-i', '--input-folder', default = '.',
                      dest='input_path', help='path to TCR Vb sequencing files')
    parser.add_option('-o', '--output-folder', default = '.',
                      dest='output_path', help='path to save results')
    parser.add_option('-d', '--delimiter', default='tab',
                      type='choice', dest='delimiter',
                      choices=['tab', 'comma'],
                      help='Declare TCR Vb file delimiter. ' \
                           'Defaults to tab. Options: ["tab", "comma"].')
    parser.add_option('-v',
                      default=None, dest='v_col',
                      help='Column name containing V gene information. ' \
                           'If not provided, it will not be used to define a TCR clone.')
    parser.add_option('-j',
                      default=None,
                      dest='j_col',
                      help='Column name containing J gene information. ' \
                           'If not provided, it will not be used to define a TCR clone.')
    parser.add_option('--cdr3', default=None,
                      dest='cdr3_col',
                      help='Column name containing CDR3 gene information. ' \
                            'If not provided, it will not be used to define a TCR clone. ' \
                            'The model is agnostic to this being nucleotide or aa sequences.')
    parser.add_option('--counts',
                      dest='counts_col',
                      help='Column name containing count information.')
    parser.add_option('--cdr3aa',
                      dest='cdr3aa_col',
                      help='Column name containing CDR3aa information. ' \
                           'Used to filter on productive sequences.')
    parser.add_option('--stat-thresh', default=0.05,
                      dest='stat_thresh',
                      help='p-value or FDR threshold to use for outlier selection.')
    parser.add_option('--pval-or-fdr', default='fdr',
                      dest='pval_or_fdr', type='choice',
                      choices=['fdr', 'pvalue'],
                      help='Whether to use p-value or FDR selection. ' \
                           'Defaults to FDR. Options: ["fdr", "pvalue"].')
    parser.add_option('--beta-param-constant', default='True',
                      dest='beta_param', type='choice',
                      choices=['True', 'False'],
                      help='Whether to keep the beta parameter constant across samples. ' \
                           'Defaults to True. We recommend leaving this to True.')
    parser.add_option('--which-QC', default='strictQC',
                      dest='which_QC', type='choice',
                      choices = ['strictQC', 'looseQC', 'noQC'],
                      help='How to select which clones are included in the analysis. ' \
                           'Options are ["strictQC", "looseQC", "noQC"]. ' \
                           'With "strictQC", keep clones if present at count>2 in >1 timepoint.' \
                           'With "looseQC", keep clones if present at count>0 in >1 timepoint.' \
                           'With "noQC", all clones are included. ' \
                           'This makes the implementation very slow and may give noisier results.' \
                           'Defaults to "strictQC".')
    parser.add_option('--transform', default='g',
                      dest='which_transform', type='choice',
                      choices=['g', 'log'],
                      help='Which transformation to apply to the frequencies. ' \
                           'Options are ["g(f)", "log"]. Defaults to "g" which implements g(f).')

    (options, args) = parser.parse_args()

    # check all needed arguments are there
    if options.input_path is None:
        parser.error('The --input-folder argument is needed.')
    if options.metadata is None:
        parser.error('The --metadata argument is needed.')
    if options.cdr3aa_col is None:
        parser.error('The --CDR3aa argument is needed to filter on productive sequences.')

    return vars(options)

def main():
    args_dict = parse_all_arguments()

    input_path = Path(args_dict['input_path'])
    output_path = Path(args_dict['output_path'])
    delimiter = args_dict['delimiter']

    columns = [args_dict['v_col'], args_dict['j_col'],
                args_dict['cdr3_col'],
                 args_dict['counts_col'], args_dict['cdr3aa_col']]
    columns = [c for c in columns if c is not None]

    clone_id_cols = [args_dict['cdr3_col'], args_dict['v_col'], args_dict['j_col']]
    clone_id_cols = [c for c in clone_id_cols if c is not None]

    metadata = pd.read_csv(args_dict['metadata'])
    sample_list = metadata['sample'].tolist()

    counts_df = load_data(input_path, output_path, delimiter, columns, clone_id_cols, sample_list)
    sample_order = metadata.sort_values(by = 'timepoint', ascending=True)['sample'].tolist()
    timepoint_dictionary = dict(zip(metadata['sample'].tolist(), metadata['timepoint'].tolist()))

    counts_df = counts_df[sample_order] # re-order to ensure same order as sample order
    all_clones = counts_df.index.tolist()
    Nr = counts_df.sum(axis=0).values
    counts = counts_df.values
    assert counts.shape == (len(all_clones), len(sample_order))

    stat_thresh = args_dict['stat_thresh']
    pval_or_fdr = args_dict['pval_or_fdr'].lower()
    which_beta = 'constantBeta' if args_dict['beta_param'] == 'True' else 'constantB'
    which_QC = args_dict['which_QC']
    which_transform = args_dict['which_transform']

    print('Calculating outliers...')
    outlier_list, pca_fit, R_thresh, X_transformed = \
                    CloneSearch(counts, Nr, all_clones,
                                sample_order, timepoint_dictionary, stat_thresh,
                                pval_or_fdr, which_beta, which_QC, which_transform)

    print(f'{len(outlier_list)} outliers identified with R threshold = {R_thresh}.')

    pca_fit.to_csv(output_path/'PCA.csv')
    X_transformed.to_csv(output_path/'transformed_frequencies.csv')
    with open(output_path/'outliers.txt', 'w') as f:
        for c in outlier_list:
            f.write(c + '\n')
    print(f'Outliers saved to {output_path}')

if __name__ == '__main__':
    main()
