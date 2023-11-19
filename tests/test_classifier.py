import csv

import pandas as pd

from bot.classification import PartsClassifier, train_model_with_features
from bot.feature_extract import extract_features_with_openai
from bot.utils import normalize_text


def test_train_model_with_feature():
    train_model_with_features("data/test1.csv", "feature1")

    classifier = PartsClassifier("feature1")

    part_df = pd.DataFrame(
        {
            "original_string": ["316L不锈钢管道增压泵 150SG140-26"],
            "type": ["增压泵"],
            "model_number": ["150SG140-26"],
            "material": ["316L不锈钢"],
            "function": [""],
            "dimension": [""],
            "extra": [""],
        }
    )

    pred_df = classifier.guess(part_df)
    pred_df.iloc[0]["prediction"] == "泵管阀"


# very poor performance, not worth keeping
# def test_alt_classify():
#     model = train_model_with_embedding("data/embedding1.csv")
#     model.save_model("data/embedding1.model")


def test_batch_predict(fromfile):
    classifier = PartsClassifier("feature1")

    with open(fromfile, "r") as input_f:
        parts = [normalize_text(line) for line in input_f.readlines() if len(line.strip()) > 3]
        feature_df = extract_features_with_openai(pd.DataFrame({"original_string": parts}), "original_string", "35t")
        prediction_df = classifier.guess(feature_df)
        assert len(prediction_df) == len(parts)


def test_extract_features():
    xls_df = pd.read_excel("data/test1.xlsx", sheet_name="输出（含人工分类结果）")

    # concat relevant column into original_string column
    test_df = xls_df[["物料描述", "备注", "一级类目"]].applymap(normalize_text)
    test_df["original_string"] = test_df["物料描述"]  # + " " + test_df["备注"]
    test_df.rename({"一级类目": "category"}, axis=1, inplace=True)
    test_df = test_df.drop(["物料描述", "备注"], axis=1)

    # should be all 0 at this point
    print("\n", test_df.isnull().sum())

    features_df = extract_features_with_openai(test_df, "original_string", "35t")
    features_df.to_csv("data/test1.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"===INFO: writing {len(features_df)} rows")


def test_predict_all():
    classifier = PartsClassifier("feature1")

    df = pd.read_excel("data/test1.xlsx", sheet_name="输出（含人工分类结果）")
    df = df[["物料描述", "一级类目"]].applymap(normalize_text)
    df.rename({"物料描述": "original_string", "一级类目": "category"}, axis=1, inplace=True)

    feature_df = extract_features_with_openai(df, "original_string", "35t")
    df["prediction"] = classifier.guess(feature_df)["prediction"]

    hit_ratio = len(df[df["category"] == df["prediction"]]) / len(df)
    print(f"hit ratio: {hit_ratio:.4f}")

    df.to_csv("data/compare1.csv", index=False)
