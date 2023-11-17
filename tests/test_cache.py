import os

from bot import cache


def test_find_extracted_features():
    original_text = "test_text"
    version = "1"
    feature_dict = {"feature1": "value1", "feature2": "value2"}
    fdir, file_name = cache.cache_file_name(original_text, version)
    cache_file = f"{fdir}/{file_name}"

    # at first it's not in cache
    assert cache.find_extracted_features(original_text, version) is None

    cache.save_extracted_feature(original_text, version, feature_dict)
    assert os.path.exists(cache_file)

    # Test find_extracted_features
    found_features = cache.find_extracted_features(original_text, version)
    assert found_features == feature_dict

    os.unlink(cache_file)
    assert cache.find_extracted_features(original_text, version) is None
