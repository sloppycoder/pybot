import hashlib
import json
import os
import sqlite3
from datetime import date
from sqlite3 import Error


def cache_dir() -> str:
    # change this if this is code is included in some package!
    return os.path.dirname(os.path.dirname(__file__)) + "/cache"


def get_connection() -> sqlite3.Connection:
    conn = None
    try:
        conn = sqlite3.connect(cache_dir() + "/cache.db")
        cursor = conn.cursor()

        # Check if table cache_index exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='features_index' ;")
        if cursor.fetchone() is None:
            # Create the table if it doesn't exist
            cursor.execute(
                """
                CREATE TABLE features_index (
                    original_text TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    location TEXT NOT NULL
                );
            """
            )
    except Error as e:
        print(e)

    return conn


def find_extracted_features(original_text: str, version: str) -> dict | None:
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM features_index WHERE original_text = ? AND version = ?", (original_text, version))
        record = cursor.fetchone()

        if record is not None:
            location = record[3]

            # Check if the file identified by location column exists
            if os.path.exists(location):
                # If the file exists, json.load the file content
                with open(location, "r") as file:
                    data = json.load(file)
                return data
            else:
                # If the file does not exist, delete the database record
                cursor.execute(
                    "DELETE FROM features_index WHERE original_text = ? AND version = ?",
                    (original_text, version),
                )
                conn.commit()

    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()

    return None


def cache_file_name(original_name: str, version: str) -> tuple[str, str]:
    """use today/<sha1 2 digits>/<sha1>.json as cache file name"""
    sha1_hash = hashlib.sha1()
    sha1_hash.update(original_name.encode("utf-8"))
    hex_digest = sha1_hash.hexdigest()

    today = date.today().strftime("%Y%m%d")

    return f"{cache_dir()}/{today}/{hex_digest[:2]}", f"{hex_digest}_{version}.json"


def save_extracted_feature(original_text: str, version: str, feature_dict: str) -> None:
    try:
        conn = get_connection()
        cursor = conn.cursor()

        # json.dump feature_dict to a file
        dirp, file_name = cache_file_name(original_text, version)
        cache_file = f"{dirp}/{file_name}"
        os.makedirs(dirp, exist_ok=True)
        with open(cache_file, "w") as file:
            json.dump(feature_dict, file)

        # Run a select on the table with original_text and version
        cursor.execute("SELECT * FROM features_index WHERE original_text = ? AND version = ?", (original_text, version))
        record = cursor.fetchone()

        if record is not None:
            # If the record exists, update the location with file name and update created_at with current timestamp
            cursor.execute(
                "UPDATE features_index SET location = ?, created_at = CURRENT_TIMESTAMP "
                + "WHERE original_text = ? AND version = ?",
                (cache_file, original_text, version),
            )
        else:
            # If the record does not exist, insert a new record
            cursor.execute(
                "INSERT INTO features_index(original_text, version, location) VALUES (?, ?, ?)",
                (original_text, version, cache_file),
            )

        conn.commit()

    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()
