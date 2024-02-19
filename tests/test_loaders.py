from tensorflow_time_series_dataset.loaders import CSVDataLoader


def test_csv_loader(tmp_csv_with_id):
    test_data_path, test_df = tmp_csv_with_id
    csv_loader = CSVDataLoader(test_data_path)
    loaded_df = csv_loader()
    assert loaded_df.equals(test_df.set_index("date_time")), "Error: CSVLoader failed"
