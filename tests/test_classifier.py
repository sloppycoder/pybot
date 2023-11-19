import csv
import os

import pandas as pd

from bot.classification import PartsClassifier, train_model_with_features
from bot.feature_extract import extract_features_with_openai
from bot.utils import normalize_text

__FILE_TYPES__ = {
    "feature": ["feature", ".csv"],
    "model": ["model", ""],
    "verify": ["compare", ".csv"],
}


def file_path(file_type: str, batch: str) -> tuple[str, str]:
    try:
        prefix, suffix = __FILE_TYPES__[file_type]
        data_dir = os.environ.get("DATA_DIR", "data")
        return f"{data_dir}/{file_type}_{batch}{suffix}"
    except KeyError:
        return "unknown", ""


def load_test_data() -> pd.DataFrame:
    xls_df = pd.read_excel("data/test1.xlsx", sheet_name="输出（含人工分类结果）")

    # concat relevant column into original_string column
    df = xls_df[["物料描述", "备注", "一级类目"]].applymap(normalize_text)
    # uncomment this to use memo column for feature extraction too
    # improvement is marginal and probably won't work for other data
    df["original_string"] = df["物料描述"]  # + " " + df["备注"]
    df.rename({"一级类目": "category"}, axis=1, inplace=True)
    df = df.drop(["物料描述", "备注"], axis=1)
    df = df.applymap(normalize_text)

    # should be all 0 at this point
    print("\n")
    print(df.isnull().sum())

    return df


def test_train_model_with_feature(batch):
    train_model_with_features(
        file_path("feature", batch),
        file_path("model", batch),
    )

    model_prefix = file_path("model", batch)
    assert os.path.exists(model_prefix + "_model.joblib")
    assert os.path.exists(model_prefix + "_pipeline.joblib")
    assert os.path.exists(model_prefix + "_encoder.joblib")

    classifier = PartsClassifier(file_path("model", batch))

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


def test_batch_predict(fromfile, batch):
    classifier = PartsClassifier(file_path("model", batch))

    with open(fromfile, "r") as input_f:
        parts = [normalize_text(line) for line in input_f.readlines() if len(line.strip()) > 3]
        feature_df = extract_features_with_openai(pd.DataFrame({"original_string": parts}), "original_string", "35t")
        prediction_df = classifier.guess(feature_df)
        assert len(prediction_df) == len(parts)


def test_extract_features(batch):
    test_df = load_test_data()
    output_file = file_path("feature", batch)

    features_df = extract_features_with_openai(test_df, "original_string", "35t")
    features_df.to_csv(file_path("feature", batch), index=False, quoting=csv.QUOTE_NONNUMERIC)
    print(f"===INFO: writing {len(features_df)} rows to {output_file}")

    assert os.path.exists(file_path("feature", batch))


def test_predict_all(batch):
    df = load_test_data()

    feature_df = extract_features_with_openai(df, "original_string", "35t")

    classifier = PartsClassifier(file_path("model", batch))
    df["prediction"] = classifier.guess(feature_df)["prediction"]

    hit_ratio = len(df[df["category"] == df["prediction"]]) / len(df)
    print(f"hit ratio: {hit_ratio:.4f}")

    df.to_csv(file_path("verify", batch), index=False)
