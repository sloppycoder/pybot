import dbm
import hashlib
import json
import logging as log
import os
from datetime import datetime

# this will be initialized the first time cache_dir() is called
__cache_dir__ = None


def cache_dir() -> str:
    global __cache_dir__
    if __cache_dir__ is None:
        __cache_dir__ = os.environ.get("CACHE_DIR", None) or "./cache"
        __cache_dir__ = os.path.abspath(__cache_dir__)
        log.info(f"cache_dir: {__cache_dir__}")
    return __cache_dir__

    # def migrate_cache() -> None:
    #     parent_dir = cache_dir()
    #     new_parent_dir = parent_dir + "_2"
    #     db = dbm.open(new_parent_dir + "/cache.db", "c")

    #     for root, dirs, nfiles in os.walk(parent_dir):
    #         for filename in nfiles:
    #             if filename.endswith(".json"):
    #                 cache_file = os.path.join(root, filename)
    #                 with open(cache_file, "r") as f:
    #                     data = json.load(f)
    #                     try:
    #                         sha1, version = tuple(os.path.basename(cache_file).split("_"))
    #                         version = version.split(".")[0]
    #                         original_string = data["original_string"]
    #                         rel_dir, file_name = cache_path(original_string, version)
    #                         if sha1 in file_name:
    #                             log.info(f"migrating cache file: {cache_file} to {rel_dir}")
    #                             os.makedirs(new_parent_dir + "/" + rel_dir, exist_ok=True)
    #                             new_filename = rel_dir + "/" + file_name
    #                             with open(new_parent_dir + "/" + new_filename, "w") as f:
    #                                 json.dump(data, f)
    #                             try:
    #                                 new_dict = json.loads(db[original_string])
    #                                 new_dict[version] = new_filename
    #                                 db[original_string] = json.dumps(new_dict)
    #                             except KeyError:
    #                                 db[original_string] = json.dumps({version: new_filename})
    #                         else:
    #                             log.info(f"skippping cache file: {cache_file}, different sha?")
    #                     except IndexError:
    #                         log.info(f"skipping cache file: {cache_file}")

    # db.close()


def get_cache_db():
    return


def find_extracted_features(key: str, version: str) -> dict | None:
    with dbm.open(cache_dir() + "/cache.db", "c") as db:
        try:
            cached = db[key]
            location = json.loads(cached)[version]
            cache_file = cache_dir() + "/" + location
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                # remove stale cache entry
                del db[key]
                return None
        except KeyError:
            return None


def cache_path(key: str, version: str) -> tuple[str, str]:
    """use yyyymmdd/HHMM/<sha1>_<version>.json as cache file name"""
    sha1_hash = hashlib.sha1()
    sha1_hash.update(key.encode("utf-8"))
    hex_digest = sha1_hash.hexdigest()

    day, hm = tuple(datetime.today().strftime("%Y%m%d-%H%M").split("-"))
    rel_dir = f"{version}/{day}/{hm}"

    return rel_dir, f"{hex_digest}_{version}.json"


def save_extracted_feature(key: str, version: str, data: dict) -> None:
    rel_dir, filename = cache_path(key, version)
    rel_file_path = rel_dir + "/" + filename

    os.makedirs(cache_dir() + "/" + rel_dir, exist_ok=True)
    with open(cache_dir() + "/" + rel_file_path, "w") as f:
        json.dump(data, f)

    with dbm.open(cache_dir() + "/cache.db", "c") as db:
        try:
            # update an existing entry
            cached = json.loads(db[key])
            cached[version] = rel_file_path
            db[key] = json.dumps(cached)
        except KeyError:
            # create new entry
            db[key] = json.dumps({version: rel_file_path})
