def test_api_predict_endpoint(app_client):
    payload = {
        'packet_size': 400,
        'bytes_sent': 1200,
        'source_ip': '192.168.1.11',
        'dest_ip': '10.0.0.33',
        'protocol': 'TCP',
        'timestamp_seconds': 1700000100,
        'hour': 9
    }
    r = app_client.post('/api/predict?threshold=0.5', json=payload)
    assert r.status_code == 200
    data = r.get_json()
    assert 'prediction' in data and data['prediction'] in (0,1)
    assert 'threshold' in data
    if 'probability' in data and data['probability'] is not None:
        assert 0.0 <= data['probability'] <= 1.0


def test_api_metrics_endpoint(app_client):
    r = app_client.get('/api/metrics')
    assert r.status_code == 200
    assert isinstance(r.get_json(), dict)
