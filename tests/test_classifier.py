import csv

import numpy as np
import pandas as pd

from bot.classification import (
    PartsClassifier,
    train_model_with_embedding,
    train_model_with_features,
)
from bot.feature_extract import InvalidResponse, extract_features_with_openai
from bot.utils import normalize_text


def test_classify():
    train_model_with_features("data/test1.csv", "feature1")


def test_alt_classify():
    model = train_model_with_embedding("data/embedding1.csv")
    model.save_model("data/embedding1.model")


def test_predict():
    classifier = PartsClassifier("feature1")

    part = pd.DataFrame(
        {
            "original_string": ["316L不锈钢管道增压泵 150SG140-26"],
            "type": ["增压泵"],
            "model_number": ["150SG140-26"],
            "material": ["316L不锈钢"],
            "function": ["不详"],
            "dimension": ["不详"],
            "brand": ["不详"],
            "model number": ["不详"],
            "component": ["不详"],
        }
    )
    pred = classifier.guess(part)
    assert pred.tolist()[0] == "泵管阀"


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
