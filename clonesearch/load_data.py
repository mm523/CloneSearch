from optparse import OptionParser
from pathlib import Path
import pandas as pd

'''
The following arguments are user specified and are needed: 
- --metadata: path to metadata file
- --cdr3aa: Column name for CDR3aa sequence information in TCRVb sequencing file. 
            Used for filtering of productive sequences.
- --counts: Column name containing count information

The following arguments can be 'None' and will determine how a TCR clonotype is defined:
- -v/--V-column: Column name containing V gene information. 
                 If not provided, it will not be used to define a TCR clone.
- -j/--J-column: Column name containing J gene information. 
                 If not provided, it will not be used to define a TCR clone.
- --cdr3: Column name containing CDR3 gene information. 
          If not provided, it will not be used to define a TCR clone. 
          The model is agnostic to this being nucleotide or aa sequences.

The following arguments can be toggled by the user:
- -d/--delimiter: Delimiter used in TCR Vb sequencing files. 
                  Defaults to tab. Options: ["tab", "comma"]
- -i/--input-folder: Path to TCR Vb sequencing files. Defaults to current folder.
- -o/--output-folder: Path to TCR Vb sequencing files. Defaults to current folder.
'''

def load_data(input_path, output_path, delimiter, columns, clone_id_cols, sample_list):

    if not input_path.exists():
        raise ValueError(f'Path {input_path} does not exist')

    if not output_path.exists():
        raise ValueError(f'Path {output_path} does not exist')

    delimiter = '\t' if delimiter =='tab' else \
                    ',' if delimiter =='comma' else \
                        'nope'
    if delimiter == 'nope':
        raise ValueError('Delimiter not recognised')

    def _check_cdr3_is_aa(cdr3):
        if len(cdr3) == 0:
            return False
        AAs = list('ARNDCEQGHILKMFPSTWYV')
        if not all(c in AAs for c in cdr3):
            return False
        return True

    samples = []
    for s in sample_list:
        c_sample = pd.read_csv(input_path/s, sep = delimiter,
                                usecols=columns)
        c_sample = c_sample.dropna(subset=clone_id_cols)
        c_sample['is_productive'] = [_check_cdr3_is_aa(cdr3) for cdr3 in c_sample[columns[-1]]]
        c_sample = c_sample.loc[c_sample['is_productive'] == True]
        c_sample['clone_id'] = c_sample[clone_id_cols].astype(str).agg('::'.join, axis=1)
        c_sample['sample'] = s
        c_sample = c_sample.rename(columns = {columns[-2]:'counts'})
        samples.append(c_sample.drop(clone_id_cols + ['is_productive', columns[-1]], axis=1))

    # merge all
    samples = pd.concat(samples)
    samples_wide = samples.pivot(
                    index = ['clone_id'],
                    columns = ['sample'],
                    values = ['counts']).fillna(0)
    samples_wide.reset_index().to_csv(output_path/'counts_all_clones.csv.gz')

    print('All data loaded and formatted into counts per TCR.')
    return samples_wide

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
    parser.add_option('-v', '--V-column',
                      default=None, dest='v_col',
                      help='Column name containing V gene information. ' \
                           'If not provided, it will not be used to define a TCR clone.')
    parser.add_option('-j', '--J-column', default=None,
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

    (options, args) = parser.parse_args()
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

    clone_id_cols = [args_dict['v_col'], args_dict['j_col'], args_dict['cdr3_col']]
    clone_id_cols = [c for c in clone_id_cols if c is not None]

    metadata = pd.read_csv(args_dict['metadata'])
    sample_list = metadata['sample'].tolist()

    load_data(input_path, output_path, delimiter, columns, clone_id_cols, sample_list)

if __name__ == '__main__':
    main()
