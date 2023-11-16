import csv

import numpy as np
import pandas as pd

from bot.classifier import (
    InvalidResponse,
    extract_features_with_openai,
    train_classification_model,
)
from bot.utils import normalize_text


def test_classify():
    train_classification_model("data/test1.csv")


def test_extract_features():
    chunk_size = 30

    full_df = pd.read_excel("data/test1.xlsx", sheet_name="输出（含人工分类结果）")
    test_df = full_df[["物料描述", "一级类目"]].applymap(normalize_text).head(1500)
    test_df.rename({"物料描述": "description", "一级类目": "category"}, axis=1, inplace=True)

    skipped = 0
    features_df = pd.DataFrame()
    for _, chunk in test_df.groupby(np.arange(len(test_df)) // chunk_size):
        try:
            df = extract_features_with_openai(chunk)
            if len(df) > 0:
                # save the result to file everytime result is returned
                # since openai can be very very slow
                features_df = pd.concat([features_df, df])
                features_df.to_csv("data/test1.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
                print(f"===INFO: writing {len(features_df)} rows")
        except InvalidResponse as e:
            print(f"===ERROR: {e}")
            skipped += len(chunk)

    if skipped:
        print("===WARN: skipped {skipped} rows")
