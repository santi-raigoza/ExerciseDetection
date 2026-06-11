import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def load_data(data_dir):
    """Load all CSVs from data_dir. Returns X (n, 2970) and y (n,) as numpy arrays."""
    csvs = list(Path(data_dir).glob('*.csv'))
    if not csvs:
        raise ValueError(f"No CSV files found in {data_dir}")

    # Each CSV has a 'label' column + 2970 feature columns (one row = one 30-frame window)
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)

    # X: the feature matrix — every column except 'label', converted to float32
    X = df.drop(columns=['label']).values.astype(np.float32)

    # y: the label array — one string per row ('pushup', 'squat', etc.)
    y = df['label'].values
    return X, y


def train(data_dir, models_dir, **mlp_kwargs):
    X, y = load_data(data_dir)

    # Hold out 20% of data for evaluation — the model never sees this during training.
    # stratify=y ensures each class is proportionally represented in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # StandardScaler shifts each feature to mean=0 and scales it to std=1.
    # Neural networks train much faster and more reliably on normalized inputs.
    # fit_transform on train data only — then apply the same scale to test data.
    # (We must NOT fit on test data — that would leak information about the test set.)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # MLPClassifier = Multi-Layer Perceptron, a feed-forward neural network.
    # hidden_layer_sizes=(256, 128) means two hidden layers: 256 neurons, then 128.
    # Input layer: 2970 features (30 frames × 99 landmark values per frame)
    # Output layer: one node per class (pushup, squat, pullup, jumping_jack, rest)
    # mlp_kwargs lets tests override these with tiny values (e.g. hidden_layer_sizes=(8,))
    # so training completes in milliseconds instead of minutes.
    params = dict(hidden_layer_sizes=(256, 128), max_iter=500, random_state=42)
    params.update(mlp_kwargs)
    model = MLPClassifier(**params)
    model.fit(X_train_s, y_train)

    # Evaluate on the held-out test set.
    # classification_report shows precision, recall, and F1 per class.
    # confusion_matrix shows how often each class was confused with another.
    y_pred = model.predict(X_test_s)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))

    # Save both artifacts: the model (decision logic) and the scaler (normalization params).
    # Both are needed at inference time — app.py loads them to classify live webcam frames.
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  Path(models_dir) / 'model.pkl')
    joblib.dump(scaler, Path(models_dir) / 'scaler.pkl')
    return model, scaler


def main():
    parser = argparse.ArgumentParser(description='Train exercise classifier')
    parser.add_argument('--data-dir',   default='data/raw')
    parser.add_argument('--models-dir', default='models')
    args = parser.parse_args()
    train(args.data_dir, args.models_dir)


if __name__ == '__main__':
    main()
