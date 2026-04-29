"""
Tests for data loading: tabular file reader, delimiter detection, fid/iid index handling.

Usage:
    pytest tests/test_data_loading.py -v
"""

import os
import json
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="torchlimix_test_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_df():
    """50 samples, 3 traits, with fid/iid columns."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        'fid': np.arange(1, n + 1),
        'iid': np.arange(1, n + 1),
        'trait_0': np.random.randn(n),
        'trait_1': np.random.randn(n),
        'trait_2': np.random.randn(n),
    })


@pytest.fixture
def sample_annot_df():
    return pd.DataFrame({
        'chrom': [1] * 20 + [2] * 20,
        'pos': list(range(100, 140)),
    })


class TestDetectDelimiter:

    def _detect(self, path):
        from torchlimix.utils.data_loader import _detect_delimiter
        return _detect_delimiter(path)

    def test_comma_csv(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.csv")
        with open(path, 'w') as f:
            f.write("a,b,c\n1,2,3\n")
        assert self._detect(path) == ','

    def test_tab_tsv(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.tsv")
        with open(path, 'w') as f:
            f.write("a\tb\tc\n1\t2\t3\n")
        assert self._detect(path) == '\t'

    def test_semicolon_txt(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.txt")
        with open(path, 'w') as f:
            f.write("a;b;c\n1;2;3\n")
        assert self._detect(path) == ';'

    def test_space_txt(self, tmp_dir):
        path = os.path.join(tmp_dir, "spaced.txt")
        with open(path, 'w') as f:
            f.write("fid iid trait1 trait2\n1 1 0.5 0.6\n")
        assert self._detect(path) == ' '

    def test_csv_with_semicolons(self, tmp_dir):
        """European-style CSV using semicolons."""
        path = os.path.join(tmp_dir, "european.csv")
        with open(path, 'w') as f:
            f.write("fid;iid;trait1;trait2\n1;1;0.5;0.6\n")
        assert self._detect(path) == ';'

    def test_csv_with_tabs(self, tmp_dir):
        """CSV file actually using tabs."""
        path = os.path.join(tmp_dir, "tabbed.csv")
        with open(path, 'w') as f:
            f.write("fid\tiid\ttrait1\ttrait2\n1\t1\t0.5\t0.6\n")
        assert self._detect(path) == '\t'

    def test_tsv_with_commas(self, tmp_dir):
        """TSV file actually using commas."""
        path = os.path.join(tmp_dir, "mislabeled.tsv")
        with open(path, 'w') as f:
            f.write("fid,iid,trait1,trait2\n1,1,0.5,0.6\n")
        assert self._detect(path) == ','

    def test_csv_tie_prefers_comma(self, tmp_dir):
        path = os.path.join(tmp_dir, "tie.csv")
        with open(path, 'w') as f:
            f.write("a,b;c,d;e\n")  # 2 commas, 2 semicolons
        assert self._detect(path) == ','

    def test_tsv_tie_prefers_tab(self, tmp_dir):
        path = os.path.join(tmp_dir, "tie.tsv")
        with open(path, 'w') as f:
            f.write("a\tb,c\td,e\n")  # 2 tabs, 2 commas
        assert self._detect(path) == '\t'

    def test_txt_tie_prefers_tab(self, tmp_dir):
        path = os.path.join(tmp_dir, "tie.txt")
        with open(path, 'w') as f:
            f.write("a\tb,c\td,e\n")
        assert self._detect(path) == '\t'

    def test_single_column_fallback(self, tmp_dir):
        path = os.path.join(tmp_dir, "single.txt")
        with open(path, 'w') as f:
            f.write("value\n1\n2\n3\n")
        assert self._detect(path) == '\t'

class TestReadTabularFile:

    def _read(self, path, **kwargs):
        from torchlimix.utils.data_loader import _read_tabular_file
        return _read_tabular_file(path, **kwargs)

    def test_csv_with_header(self, tmp_dir, sample_df):
        path = os.path.join(tmp_dir, "pheno.csv")
        sample_df.to_csv(path, index=False)
        df = self._read(path)
        assert len(df) == 50

    def test_csv_no_header(self, tmp_dir, sample_df):
        path = os.path.join(tmp_dir, "pheno_noheader.csv")
        sample_df.to_csv(path, index=False, header=False)
        df = self._read(path)
        assert len(df) == 50
        assert isinstance(df.columns[0], int) or df.columns[0] == 0

    def test_csv_semicolon_end_to_end(self, tmp_dir, sample_df):
        """Semicolon-delimited .csv should read correctly."""
        path = os.path.join(tmp_dir, "pheno_semi.csv")
        sample_df.to_csv(path, sep=';', index=False)
        df = self._read(path)
        assert len(df) == 50
        assert df.shape[1] >= 4

    def test_tsv_with_header(self, tmp_dir, sample_df):
        path = os.path.join(tmp_dir, "pheno.tsv")
        sample_df.to_csv(path, sep='\t', index=False)
        df = self._read(path)
        assert len(df) == 50

    def test_txt_tab_delimited(self, tmp_dir, sample_df):
        path = os.path.join(tmp_dir, "pheno.txt")
        sample_df.to_csv(path, sep='\t', index=False)
        df = self._read(path)
        assert len(df) == 50

    def test_txt_semicolon_delimited(self, tmp_dir, sample_df):
        path = os.path.join(tmp_dir, "pheno_semi.txt")
        sample_df.to_csv(path, sep=';', index=False)
        df = self._read(path)
        assert len(df) == 50

    def test_xlsx(self, tmp_dir, sample_df):
        path = os.path.join(tmp_dir, "pheno.xlsx")
        sample_df.to_excel(path, index=False)
        df = self._read(path)
        assert len(df) == 50

    def test_unsupported_extension(self, tmp_dir):
        path = os.path.join(tmp_dir, "data.json")
        with open(path, 'w') as f:
            json.dump({"a": 1}, f)
        from torchlimix.utils.data_loader import _read_tabular_file
        with pytest.raises(ValueError, match="Unsupported file extension"):
            _read_tabular_file(path)

    def test_same_data_across_formats(self, tmp_dir, sample_df):
        csv_path = os.path.join(tmp_dir, "data.csv")
        tsv_path = os.path.join(tmp_dir, "data.tsv")
        xlsx_path = os.path.join(tmp_dir, "data.xlsx")

        sample_df.to_csv(csv_path, index=False)
        sample_df.to_csv(tsv_path, sep='\t', index=False)
        sample_df.to_excel(xlsx_path, index=False)

        df_csv = self._read(csv_path)
        df_tsv = self._read(tsv_path)
        df_xlsx = self._read(xlsx_path)

        np.testing.assert_array_almost_equal(
            df_csv.values.astype(float), df_tsv.values.astype(float)
        )
        np.testing.assert_array_almost_equal(
            df_csv.values.astype(float), df_xlsx.values.astype(float), decimal=10
        )


class TestSetFidIidIndex:

    @staticmethod
    def _index(df):
        from torchlimix.utils.data_loader import MultitaskDatasetSNP
        return MultitaskDatasetSNP._set_fid_iid_index(df)

    def test_named_fid_iid(self):
        df = pd.DataFrame({'fid': [1, 2], 'iid': [1, 2], 'val': [0.5, 0.6]})
        result = self._index(df)
        assert isinstance(result.index, pd.MultiIndex)
        assert list(result.index.names) == ['fid', 'iid']

    def test_case_insensitive(self):
        df = pd.DataFrame({'FID': [1, 2], 'IID': [1, 2], 'val': [0.5, 0.6]})
        result = self._index(df)
        assert list(result.index.names) == ['fid', 'iid']

    def test_fallback_first_two_columns(self):
        df = pd.DataFrame({'sample': [1, 2], 'replicate': [1, 2], 'val': [0.5, 0.6]})
        result = self._index(df)
        assert isinstance(result.index, pd.MultiIndex)
        assert list(result.index.names) == ['fid', 'iid']

    def test_already_multiindex_unchanged(self):
        idx = pd.MultiIndex.from_tuples([(1, 1), (2, 2)], names=['fid', 'iid'])
        df = pd.DataFrame({'val': [0.5, 0.6]}, index=idx)
        result = self._index(df)
        assert len(result) == 2

    def test_string_to_int_conversion(self):
        df = pd.DataFrame({'fid': ['1', '2'], 'iid': ['10', '20'], 'val': [0.5, 0.6]})
        result = self._index(df)
        assert result.index.get_level_values('fid').dtype == int

    def test_preserves_data_columns(self):
        df = pd.DataFrame({
            'fid': [1, 2, 3], 'iid': [1, 2, 3],
            'PC1': [0.1, 0.2, 0.3], 'PC2': [0.4, 0.5, 0.6]
        })
        result = self._index(df)
        assert 'PC1' in result.columns
        assert 'PC2' in result.columns
        assert 'fid' not in result.columns

class TestAnnotationLoading:

    def test_csv(self, tmp_dir, sample_annot_df):
        path = os.path.join(tmp_dir, "annot.csv")
        sample_annot_df.to_csv(path, index=False)

        from torchlimix.utils.data_loader import _read_tabular_file
        df = _read_tabular_file(path)
        assert 'chrom' in df.columns and 'pos' in df.columns
        assert len(df) == 40

    def test_xlsx(self, tmp_dir, sample_annot_df):
        path = os.path.join(tmp_dir, "annot.xlsx")
        sample_annot_df.to_excel(path, index=False)

        from torchlimix.utils.data_loader import _read_tabular_file
        df = _read_tabular_file(path)
        assert 'chrom' in df.columns
        assert len(df) == 40

    def test_tsv(self, tmp_dir, sample_annot_df):
        path = os.path.join(tmp_dir, "annot.tsv")
        sample_annot_df.to_csv(path, sep='\t', index=False)

        from torchlimix.utils.data_loader import _read_tabular_file
        df = _read_tabular_file(path)
        assert 'chrom' in df.columns
        assert len(df) == 40