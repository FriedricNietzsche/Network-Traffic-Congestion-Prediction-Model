import os
import pytest
import pandas as pd
from datetime import datetime


@pytest.fixture(scope="session")
def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


@pytest.fixture(scope="session")
def quick_model(project_root):
    """Create a lightweight model & artifacts for fast tests.
    Avoids full hyperparameter search (train_model) to keep CI quick.
    """
    import numpy as np  # noqa: F401
    from src.data.generate_data import generate_traffic_data
    from src.features.feature_engineering import engineer_features
    from sklearn.ensemble import RandomForestClassifier
    import yaml, pickle

    # Ensure models dir exists
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Load base config for paths
    with open(os.path.join(project_root, "config", "config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    model_path = os.path.join(project_root, cfg['paths']['model_path'])
    features_path = os.path.join(project_root, cfg['paths']['features_path'])
    scaler_path = os.path.join(project_root, cfg['paths']['scaler_path'])

    # If model already exists, reuse to speed up local cycles
    if os.path.exists(model_path) and os.path.exists(features_path):
        yield {
            'model_path': model_path,
            'features_path': features_path,
            'scaler_path': scaler_path
        }
        return

    # Generate small synthetic dataset
    df = generate_traffic_data(n_samples=60, output_path=os.path.join(project_root, 'data', 'traffic_data_small.csv'))
    processed = engineer_features(df)  # training=True by default creates scaler
    X = processed.drop('congestion', axis=1)
    y = processed['congestion']

    # Train a very small model
    rf = RandomForestClassifier(n_estimators=25, random_state=42, class_weight='balanced')
    rf.fit(X, y)

    # Persist artifacts
    with open(model_path, 'wb') as f:
        pickle.dump(rf, f)
    X.to_pickle(features_path)

    yield {
        'model_path': model_path,
        'features_path': features_path,
        'scaler_path': scaler_path
    }


@pytest.fixture()
def app_client(quick_model):  # ensures model exists first
    from app import app as flask_app
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as client:
        yield client
