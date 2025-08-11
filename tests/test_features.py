def test_feature_engineering_shapes(tmp_path):
    from src.data.generate_data import generate_traffic_data
    from src.features.feature_engineering import engineer_features

    df = generate_traffic_data(n_samples=30, output_path=str(tmp_path / 'temp.csv'))
    processed = engineer_features(df, scaler_path='models/test_scaler.pkl')
    # Ensure target still present
    assert 'congestion' in processed.columns
    # Ensure cyclical features created
    assert {'hour_sin','hour_cos'}.issubset(set(processed.columns))
    # Ensure protocol one-hot columns exist for at least one protocol
    assert any(c.startswith('protocol_') for c in processed.columns)
