import numpy as np
import pandas as pd
import joblib
import pytest
from train import load_data, train


def _write_csv(path, label, n=20, val=1.0):
    data = {'label': [label] * n}
    data.update({f'feat_{i}': [val + np.random.randn() * 0.01] * n for i in range(2970)})
    pd.DataFrame(data).to_csv(path, index=False)


def test_load_data_returns_correct_shape(tmp_path):
    _write_csv(tmp_path / 'pushup_test.csv', 'pushup', n=10)
    X, y = load_data(tmp_path)
    assert X.shape == (10, 2970)
    assert y.shape == (10,)
    assert (y == 'pushup').all()


def test_load_data_merges_multiple_csvs(tmp_path):
    _write_csv(tmp_path / 'pushup_test.csv', 'pushup', n=10)
    _write_csv(tmp_path / 'rest_test.csv', 'rest', n=10)
    X, y = load_data(tmp_path)
    assert X.shape == (20, 2970)
    assert set(y) == {'pushup', 'rest'}


def test_load_data_raises_on_empty_dir(tmp_path):
    with pytest.raises(ValueError, match="No CSV files"):
        load_data(tmp_path)


def test_train_creates_model_and_scaler_files(tmp_path):
    data_dir = tmp_path / 'raw'
    data_dir.mkdir()
    models_dir = tmp_path / 'models'
    _write_csv(data_dir / 'pushup.csv', 'pushup', n=20, val=1.0)
    _write_csv(data_dir / 'rest.csv',   'rest',   n=20, val=-1.0)
    train(data_dir, models_dir, hidden_layer_sizes=(8,), max_iter=50)
    assert (models_dir / 'model.pkl').exists()
    assert (models_dir / 'scaler.pkl').exists()


def test_trained_model_predicts_separable_classes(tmp_path):
    data_dir = tmp_path / 'raw'
    data_dir.mkdir()
    models_dir = tmp_path / 'models'
    # Clearly separable: pushup=all 1s, rest=all -1s
    _write_csv(data_dir / 'pushup.csv', 'pushup', n=30, val=1.0)
    _write_csv(data_dir / 'rest.csv',   'rest',   n=30, val=-1.0)
    train(data_dir, models_dir, hidden_layer_sizes=(8,), max_iter=100)

    model = joblib.load(models_dir / 'model.pkl')
    scaler = joblib.load(models_dir / 'scaler.pkl')

    X_pushup = scaler.transform([np.ones(2970)])
    X_rest   = scaler.transform([-np.ones(2970)])
    assert model.predict(X_pushup)[0] == 'pushup'
    assert model.predict(X_rest)[0] == 'rest'
