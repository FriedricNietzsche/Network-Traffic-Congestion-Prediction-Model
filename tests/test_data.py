def test_generate_data_columns():
    from src.data.generate_data import generate_traffic_data
    df = generate_traffic_data(n_samples=10, output_path='data/test_generated.csv')
    expected = {"timestamp","source_ip","dest_ip","protocol","packet_size","bytes_sent","congestion"}
    assert expected.issubset(df.columns)
    assert len(df) == 10
