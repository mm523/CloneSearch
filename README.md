# CloneSearch: unbiased identification of responding T cell clones from longitudinal TCR sequencing

CloneSearch provides an easy platform for analysis and clustering of timeseries data. 

### Terms of use and disclaimers

Please consult the [Terms of Use](TermsOfUse.md) before using this code. There is a pending patent application associated with this work; the patent application number is 63/972100.

### Using CloneSearch

The key functions are in the `functions` folder.

- `CloneSearch.py` contains the main function: it takes as input the counts from the timeseries for a list of TCRs and outputs the identified responding TCRs. It relies on the other functions in the folder.
- `noise_functions.py` contains functions used to calculate the noise profiles from timeseries data
- `get_params.py` contains the functions used to extract the noise parameters from timeseries data, which are used to calculate the $g(f)$ transformation
- `normalisation_functions.py` contains the functions which transform the timeseries to be fed into the PCA. The `noise_by_size_norm_linf` function is the $g(f)$ transformation
- `gaussian_outliers.py` contains the functions which define the sphere in PCA space for the outlier selection

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

To verify the installation worked:

```
python -c "from clonesearch import CloneSearch; print('OK')"
```