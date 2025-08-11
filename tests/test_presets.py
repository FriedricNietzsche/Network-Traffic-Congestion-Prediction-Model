def test_prediction_presets_probabilities(app_client):
    """Validate API returns sane outputs for UI preset-style inputs.

    Assertions:
      - Each preset returns status 200 and valid prediction field.
      - Probability (if present) is 0..1 and consistent with threshold rule.
      - (Best effort) Peak scenario probability tends to exceed Low scenario; only enforced if margin >= 0.05 to avoid flakiness.
    """
    presets = {
        'low':   {'packet_size':150,'bytes_sent':300, 'source_ip':'192.168.0.10','dest_ip':'192.168.0.20','protocol':'TCP','hour':3},
        'peak':  {'packet_size':1300,'bytes_sent':5200,'source_ip':'10.0.0.5','dest_ip':'10.0.0.55','protocol':'TCP','hour':18},
        'burst': {'packet_size':700,'bytes_sent':3500,'source_ip':'172.16.1.2','dest_ip':'172.16.1.200','protocol':'UDP','hour':12},
        'http':  {'packet_size':1100,'bytes_sent':4400,'source_ip':'192.168.1.100','dest_ip':'192.168.1.210','protocol':'HTTP','hour':17},
    }
    probs = {}
    threshold = 0.5
    import time
    now_ts = int(time.time())
    for name, payload in presets.items():
        payload = dict(payload)  # copy
        payload['timestamp_seconds'] = now_ts
        r = app_client.post(f"/api/predict?threshold={threshold}", json=payload)
        assert r.status_code == 200, f"Preset {name} failed with status {r.status_code}"
        data = r.get_json()
        assert 'prediction' in data
        pred = data['prediction']
        assert pred in (0,1)
        proba = data.get('probability')
        if proba is not None:
            assert 0.0 <= proba <= 1.0
            # consistency with threshold rule
            if proba >= threshold:
                assert pred == 1
            else:
                assert pred == 0
            probs[name] = proba
    # Comparative heuristic: peak should usually exceed low
    if 'peak' in probs and 'low' in probs:
        diff = probs['peak'] - probs['low']
        # Only assert ordering if there's a meaningful gap to avoid random-failure flakiness
        if abs(diff) >= 0.05:
            assert probs['peak'] > probs['low'], f"Expected peak prob > low prob when gap>=0.05 (got diff={diff:.3f})"
