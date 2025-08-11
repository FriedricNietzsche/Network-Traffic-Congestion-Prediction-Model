def test_predict_function_basic(quick_model):
    from src.models.predict import predict_congestion
    sample = {
        'packet_size': 500,
        'bytes_sent': 1500,
        'source_ip': '192.168.1.10',
        'dest_ip': '10.0.0.20',
        'protocol': 'TCP',
        'timestamp_seconds': 1700000000,
        'hour': 12
    }
    pred, proba = predict_congestion(sample, return_proba=True)
    assert pred in (0,1)
    assert 0.0 <= proba <= 1.0
