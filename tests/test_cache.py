import dbm
import json
import os

from bot import cache


def test_cache_extracted_features():
    original_text = "test_text"
    version = "1"
    feature_dict = {"feature1": "value1", "feature2": "value2"}
    rel_dir, file_name = cache.cache_path(original_text, version)
    cache_file = f"{cache.cache_dir()}/{rel_dir}/{file_name}"

    # at first it's not in cache
    assert cache.find_extracted_features(original_text, version) is None

    # it will be in cache after calling save
    cache.save_extracted_feature(original_text, version, feature_dict)
    assert os.path.exists(cache_file)
    with dbm.open(cache.cache_dir() + "/cache.db", "r") as cache_db:
        assert original_text in cache_db

    # save the same feature again should only result in one cache entry
    cache.save_extracted_feature(original_text, version, feature_dict)
    assert os.path.exists(cache_file)
    with dbm.open(cache.cache_dir() + "/cache.db", "r") as cache_db:
        assert len(json.loads(cache_db[original_text])) == 1

    # find should return the result when it's in cache
    found_features = cache.find_extracted_features(original_text, version)
    assert found_features == feature_dict

    # delete the file will trigger delete stale cache entry
    os.unlink(cache_file)
    assert cache.find_extracted_features(original_text, version) is None
    with dbm.open(cache.cache_dir() + "/cache.db", "r") as cache_db:
        assert original_text not in cache_db
