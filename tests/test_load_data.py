"""
Tests for load_data(), focused on how `clone_id_cols` interacts with the
internal groupby:

- If clone_id_cols produce UNIQUE clone_ids within a sample file, no rows
  should be merged -> each row's counts pass through unchanged.
- If clone_id_cols produce NON-UNIQUE (duplicate) clone_ids within a
  sample file, those rows should be grouped together and their counts
  summed.
"""

import pandas as pd
import pytest
from clonesearch.load_data import load_data


# columns[-2] is treated as the counts column, columns[-1] as the cdr3 aa
# column used for the productivity filter. clone_id_cols is a subset of
# the other columns used to build the `clone_id`.
COLUMNS = ['v_gene', 'j_gene', 'cdr3_nt', 'duplicate_count', 'cdr3_aa']
CLONE_ID_COLS = ['v_gene', 'j_gene', 'cdr3_nt']

def _write_csv(path, rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(path, sep=',', index=False)

def test_unique_clone_id_cols_not_grouped(tmp_path):
    """
    Every row has a distinct (v_gene, j_gene, cdr3_nt) combination, so
    every row maps to a unique clone_id. The groupby should therefore
    be a no-op aggregation-wise: counts should be unchanged and no rows
    should be merged.
    """
    rows = [
        ('TRBV1', 'TRBJ1', 'ACGT', 5, 'CASSL'),
        ('TRBV2', 'TRBJ2', 'TGCA', 3, 'CASSF'),
        ('TRBV3', 'TRBJ3', 'GGCC', 7, 'CASSY'),
    ]
    _write_csv(tmp_path / 'sample1.csv', rows)

    result = load_data(
        input_path=tmp_path,
        delimiter='comma',
        columns=COLUMNS,
        clone_id_cols=CLONE_ID_COLS,
        sample_list=['sample1.csv'],
    )

    expected_counts = {
        'TRBV1::TRBJ1::ACGT': 5,
        'TRBV2::TRBJ2::TGCA': 3,
        'TRBV3::TRBJ3::GGCC': 7,
    }

    # one row per input row -> nothing got merged
    assert len(result) == 3
    for clone_id, expected in expected_counts.items():
        assert result.loc[clone_id, 'sample1.csv'] == expected


def test_non_unique_clone_id_cols_are_grouped(tmp_path):
    """
    Two rows share the same (v_gene, j_gene, cdr3_nt) -> same clone_id,
    but differ in duplicate_count and cdr3_aa. Those two rows should be
    collapsed into a single clone_id row with summed counts.
    """
    rows = [
        ('TRBV1', 'TRBJ1', 'ACGT', 5, 'CASSL'),
        ('TRBV1', 'TRBJ1', 'ACGT', 10, 'CASSG'),  # duplicate clone_id
        ('TRBV2', 'TRBJ2', 'TGCA', 3, 'CASSF'),
    ]
    _write_csv(tmp_path / 'sample2.csv', rows)

    result = load_data(
        input_path=tmp_path,
        delimiter='comma',
        columns=COLUMNS,
        clone_id_cols=CLONE_ID_COLS,
        sample_list=['sample2.csv'],
    )

    # 3 input rows -> only 2 distinct clone_ids after grouping
    assert len(result) == 2
    assert result.loc['TRBV1::TRBJ1::ACGT', 'sample2.csv'] == 15
    assert result.loc['TRBV2::TRBJ2::TGCA', 'sample2.csv'] == 3


def test_mixed_unique_and_non_unique_within_same_sample(tmp_path):
    """
    Sanity check combining both cases in one file: some clone_ids are
    unique (pass through unchanged) while others repeat (get summed),
    all within the same sample.
    """
    rows = [
        ('TRBV1', 'TRBJ1', 'ACGT', 5, 'CASSL'),
        ('TRBV1', 'TRBJ1', 'ACGT', 10, 'CASSG'),  # duplicate -> sums with row above
        ('TRBV2', 'TRBJ2', 'TGCA', 3, 'CASSF'),   # unique
        ('TRBV3', 'TRBJ3', 'GGCC', 7, 'CASSY'),   # unique
    ]
    _write_csv(tmp_path / 'sample3.csv', rows)

    result = load_data(
        input_path=tmp_path,
        delimiter='comma',
        columns=COLUMNS,
        clone_id_cols=CLONE_ID_COLS,
        sample_list=['sample3.csv'],
    )

    assert len(result) == 3
    assert result.loc['TRBV1::TRBJ1::ACGT', 'sample3.csv'] == 15
    assert result.loc['TRBV2::TRBJ2::TGCA', 'sample3.csv'] == 3
    assert result.loc['TRBV3::TRBJ3::GGCC', 'sample3.csv'] == 7


def test_narrower_clone_id_cols_reveals_grouping(tmp_path):
    """
    Rows are unique when identified by the full (v_gene, j_gene, cdr3_nt)
    triple, but two of them share the same cdr3_nt. If clone_id_cols is
    narrowed to just ['cdr3_nt'], those two rows should now be grouped
    together and summed, even though they'd have stayed separate under
    the wider clone_id_cols.
    """
    rows = [
        ('TRBV1', 'TRBJ1', 'ACGT', 5, 'CASSL'),
        ('TRBV2', 'TRBJ2', 'ACGT', 10, 'CASSG'),  # different v/j, same cdr3_nt
        ('TRBV3', 'TRBJ3', 'GGCC', 7, 'CASSY'),
    ]
    _write_csv(tmp_path / 'sample4.csv', rows)
 
    # With the full triple as clone_id_cols, all three rows are unique.
    result_full = load_data(
        input_path=tmp_path,
        delimiter='comma',
        columns=COLUMNS,
        clone_id_cols=CLONE_ID_COLS,
        sample_list=['sample4.csv'],
    )
    assert len(result_full) == 3
    assert result_full.loc['TRBV1::TRBJ1::ACGT', 'sample4.csv'] == 5
    assert result_full.loc['TRBV2::TRBJ2::ACGT', 'sample4.csv'] == 10
    assert result_full.loc['TRBV3::TRBJ3::GGCC', 'sample4.csv'] == 7
 
    # With clone_id_cols narrowed to just cdr3_nt, the first two rows
    # collapse into a single 'ACGT' clone_id with summed counts.
    result_narrow = load_data(
        input_path=tmp_path,
        delimiter='comma',
        columns=COLUMNS,
        clone_id_cols=['cdr3_nt'],
        sample_list=['sample4.csv'],
    )
    assert len(result_narrow) == 2
    assert result_narrow.loc['ACGT', 'sample4.csv'] == 15
    assert result_narrow.loc['GGCC', 'sample4.csv'] == 7


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))