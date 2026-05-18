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

### Running CloneSearch

CloneSearch has two main functions:
1. The **outlier identification** function identifies which TCRs from a dataset behave in an unexpected way (``the outliers'')
2. The **clustering** function computes trajectory similarity on a set of sequences and groups them according to similar dynamics

The suggested pipeline is to identify outliers from a longitudinal sequencing dataset, and then cluster them to identify functional groups. However, we separate these two steps so that they can be carried out on different sets of TCRs as desired by the user. 

CloneSearch can be used as a [command line tool](#command-line), or [imported into Python](#python-import).

#### Python import

Two tutorials are provided to run CloneSearch in a Jupyter notebook in the `examples` folder:
1. [`tutorial.ipynb`](examples/tutorial.ipynb) uses the key functions to show end-to-end use of CloneSearch for identification and clustering of outliers
2. [`tutorial_advanced.ipynb`](examples/tutorial_advanced.ipynb) breaks down the two key functions into step to plot the diagnostic plots to understand the inner workings of the CloneSearch function and parameter fitting to the noise profile.

The key functions are in the `clonesearch.CloneSearch` folder.

- `CloneSearch` takes as input the counts from a timeseries as an array for a list of TCRs and outputs the identified responding TCRs.
- `CloneSearch_clustering` performs clustering of a set of TCR sequences, given the trajectories.

Other functions internal to CloneSearch can be found in `clonesearch.utils` and are used for noise estimation.
- `clonsearch.utils.noise_functions` contains functions used to calculate the noise profiles from timeseries data.
- `clonsearch.utils.get_params` contains the functions used to extract the noise parameters from timeseries data, which are used to calculate the $g(f)$ transformation.
- `clonsearch.utils.normalisation_functions` contains the functions which transform the timeseries to be fed into the PCA. The `clonsearch.utils.normalisation_functions.g_of_f` function performs the $g(f)$ transformation.
- `clonsearch.utils.gaussian_outliers` contains the functions which define the sphere in PCA space for outlier selection.

We also provide a `clonesearch.plotting` module which contains some of the core plots used for diagnostics.


#### Command line

The **outlier identification** can be run with command `clonesearch-find_outliers`. The command loads TCR Vb samples as specified in a metadata file and then processes the samples to find outlier TCR trajectories. 
It outputs 3 files: 
1. a .csv file with transformed sequences for each TCR in each timepoint,
2. a .csv file with the PCA projections for each TCR,
3. a list of TCR outliers, given the identifiers provided.

The following arguments are user specified and are needed: 
- `--metadata`: Path to metadata file.
- `--cdr3aa`: Column name for CDR3aa sequence information in TCRVb sequencing file. Used for filtering of productive sequences.
- `--counts`: Column name containing count information

The following arguments can be 'None'. A TCR clone will be defined by the combination of these elements provided:
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

An example command would be:
```
clonesearch-find_outliers --metadata test_data/input/metadata.txt -i test_data/input -v bestVGene -j bestJGene --cdr3 clonalSequence --counts cloneCount --cdr3aa aaSeqCDR3 -o test_data/output
```

The **clustering** is run with `clonesearch-clustering`, which takes a list of TCRs to cluster and their trajectory information and performs clustering. It returns a .csv with the clusters indicated and a heatmap of the trajectories of TCRs in each cluster. 

`clonesearch-clustering` is agnostic on the input, but we recommend using the `transformed_frequencies.csv` and the `outliers.txt` outputs from `clonesearch-find_outliers` as inputs to the clustering function. The similarity threshold for defining clusters may need to be tuned based on user interest and number of TCRs clustered. We would not recommend running the clustering step on full repertoires without pre-selection as the computational demand increases.

The following arguments are user specified and are needed: 
- `-i/--input`: Path to file to use for calculating distances. It can be any .csv file of frequencies, transformed frequencies or PCA projections.

The following arguments can be toggled by the user:
- `--clone-list`: List of clones to cluster. If "all", all rows of the input will be clustered. Defaults to "all". If provided, row names of input_df must correspond to clone names.
- `--normalise`: Whether to normalise each row in the input by the max. Defaults to True. Recommended if using transformed frequencies.
- `--title`: The title of the colorbar. Default is "$g(f) - g(f_{\text{max}})$". It can be toggled by the user. Only used for plotting purposes.
- `--distance-metric`: Distance metric to use for clustering. 'Passed to scipy.spatial.distance.pdist. For transformed and normalised frequencies, we recommend "correlation". Defaults to "correlation".
- `--linkage-method`: Linkage method to use for clustering. Passed to scipy.cluster.hierarchy.linkage. For transformed and normalised frequencies, we recommend "average". Defaults to "average".
- `--distance-threshold`: Distance threshold to use to define clusters. Passed to scipy.cluster.hierarchy.fcluster. Defaults to 0.61.
- `-o/--output-folder`: Path to save results. Defaults to current folder.

An example command would be:
```
clonesearch-clustering --clone-list test_data/output/outliers.txt --input test_data/output/transformed_frequencies.csv --normalise True
```