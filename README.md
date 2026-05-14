# CloneSearch: unbiased identification of responding T cell clones from longitudinal TCR sequencing

CloneSearch provides an easy platform for analysis and clustering of timeseries data. 

### Terms of use and disclaimers

Please consult the [Terms of Use](TermsOfUse.md) before using this code. There is a pending patent application associated with this work; the patent application number is 63/972100.

### Installation

#### Installation from pip

#### Installation from GitHub

Clone the repository and install in editable mode:

```
git clone https://github.com/mm523/CloneSearch
cd CloneSearch
pip install -e .
```
This installs the package locally so you can import it from anywhere in your Python environment, and any changes you make to the source are reflected immediately without reinstalling.

To verify the installation worked you can run:

```
clonesearch --version
```

### Running CloneSearch for outlier identification

#### Command line

The full pipeline can be run with command `clonesearch-find_outliers`. 
The command loads TCR Vb samples as specified in a metadata file and then processes the samples to find outlier TCR trajectories. 
It outputs 3 files: 
1. a .csv file with transformed sequences for each TCR in each timepoint,
2. a .csv file with the PCA projections for each TCR,
3. a list of TCR outliers, given the identifiers provided.

The following arguments are user specified and are needed: 
- `--metadata`: Path to metadata file.
- `--cdr3aa`: Column name for CDR3aa sequence information in TCRVb sequencing file. Used for filtering of productive sequences.
- `--counts`: Column name containing count information

The following arguments can be 'None' and will determine how a TCR clonotype is defined:
- `-v`: Column name containing V gene information. If not provided, it will not be used to define a TCR clone.
- `-j`: Column name containing J gene information. If not provided, it will not be used to define a TCR clone.
- `--cdr3`: Column name containing CDR3 gene information. If not provided, it will not be used to define a TCR clone. The model is agnostic to this being nucleotide or aa sequences.

The following arguments can be toggled by the user:
- `-d/--delimiter`: Delimiter used in TCR Vb sequencing files. Defaults to tab. Options: `["tab", "comma"]`
- `-i/--input-folder`: Path to TCR Vb sequencing files. Defaults to current folder.
- `-o/--output-folder`: Path to TCR Vb sequencing files. Defaults to current folder.
- `--stat-thresh`: p-value or FDR threshold to use for outlier selection.
- `--pval-or-fdr`: Whether to use p-value or FDR selection. Defaults to FDR. Options: `["fdr", "pvalue"]`.
- `--beta-param-constant`: Whether to keep the beta parameter constant across samples. Defaults to `True`. We recommend leaving this to `True`.
- `--which-QC`: How to select which clones are included in the analysis. 
              Options are `["strictQC", "looseQC", "noQC"]`. With `strictQC`, clones are kept if present at count > 2 in > 1 timepoint. With `looseQC`, clones are kept if present at count > 0 in > 1 timepoint. With `noQC`, all clones are included. This makes the implementation very slow and may give noisier results.
              Defaults to `strictQC`.
- `--transform`: Which transformation to apply to the frequencies. Options are `["g", "log"]`. Defaults to `g`, which implements $g(f)$.

#### Python API

The key functions are in the `functions` folder.

- `clonesearch.CloneSearch.CloneSearch` is the main function: it takes as input the counts from a timeseries as an array for a list of TCRs and outputs the identified responding TCRs.
- `clonesearch.CloneSearch.CloneSearch_clustering` performs clustering of a set of TCR sequences, given the trajectories.
- `clonsearch.utils.noise_functions` contains functions used to calculate the noise profiles from timeseries data.
- `clonsearch.utils.get_params` contains the functions used to extract the noise parameters from timeseries data, which are used to calculate the $g(f)$ transformation.
- `clonsearch.utils.normalisation_functions` contains the functions which transform the timeseries to be fed into the PCA. The `clonsearch.utils.normalisation_functions.g_of_f` function performs the $g(f)$ transformation.
- `clonsearch.utils.gaussian_outliers` contains the functions which define the sphere in PCA space for outlier selection.

A sample Jupyter notebook with all the steps for outlier selection is provided in XXXX.