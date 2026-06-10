import numpy as np
import pandas as pd
import pytest
from collect import save_windows


def test_save_windows_creates_csv_with_correct_shape(tmp_path):
    windows = [np.ones(2970), np.zeros(2970)]
    save_windows(windows, 'pushup', tmp_path)
    files = list(tmp_path.glob('pushup_*.csv'))
    assert len(files) == 1
    df = pd.read_csv(files[0])
    assert df.shape == (2, 2971)   # 2970 features + 1 label column
    assert list(df.columns[:2]) == ['label', 'feat_0']
    assert (df['label'] == 'pushup').all()


def test_save_windows_values_are_correct(tmp_path):
    windows = [np.ones(2970) * 3.14]
    save_windows(windows, 'squat', tmp_path)
    df = pd.read_csv(list(tmp_path.glob('squat_*.csv'))[0])
    np.testing.assert_allclose(df.iloc[0, 1:].values.astype(float), 3.14, rtol=1e-5)


def test_save_windows_creates_output_dir_if_missing(tmp_path):
    subdir = tmp_path / 'new' / 'nested'
    windows = [np.ones(2970)]
    save_windows(windows, 'rest', subdir)
    assert subdir.exists()
    assert len(list(subdir.glob('rest_*.csv'))) == 1
