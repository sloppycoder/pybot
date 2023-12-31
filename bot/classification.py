import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

from bot import log
from bot.utils import _FEATURE_COLS_, _TARGET_COL_, blank_filler, chinese_tokenizer

_ALL_FIELDS_ = _FEATURE_COLS_ + [_TARGET_COL_]


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):  # noqa: VNE001
        return self

    def transform(self, X):  # noqa: VNE001
        return X[self.key]


# Combine the text columns using FeatureUnion
combined_features = FeatureUnion(
    [
        (
            col,
            Pipeline(
                [
                    ("selector", TextSelector(key=col)),
                    ("tfidf", TfidfVectorizer(tokenizer=chinese_tokenizer)),
                ]
            ),
        )
        for col in _FEATURE_COLS_
    ]
)


class PartsClassifier:
    def __init__(self, model_prefix: str):
        self.model = joblib.load(f"{model_prefix}_model.joblib")
        self.combined_features = joblib.load(f"{model_prefix}_pipeline.joblib")
        self.encoder = joblib.load(f"{model_prefix}_encoder.joblib")

    def guess(self, parts: pd.DataFrame) -> pd.DataFrame:
        unwanted_cols = [col for col in parts.columns if col not in _FEATURE_COLS_]
        prediction = parts.drop(unwanted_cols, axis=1).applymap(blank_filler)

        model_input = self.combined_features.transform(prediction)
        encoded_prediction = self.model.predict(model_input)
        prediction["prediction"] = self.encoder.inverse_transform(encoded_prediction)

        return prediction


def train_model_with_features(input_file: str, model_prefix: str):
    data = pd.read_csv(
        input_file,
        encoding="utf-8",
        dtype={field: str for field in _ALL_FIELDS_},  # force all fields to be string
    )

    # drop columsn we're not interested in
    data = data.drop([col for col in data.columns if col not in _ALL_FIELDS_], axis=1)

    # some category have very few samples, remove them
    class_counts = data.groupby(_TARGET_COL_)[_TARGET_COL_].count()
    bad_categories = class_counts[class_counts < 5].index.tolist()
    data = data[~data[_TARGET_COL_].isin(bad_categories)]

    # Check for missing values
    log.info(f"\n{data.isnull().sum()}")

    text_imputer = SimpleImputer(strategy="constant", fill_value="")
    data[_ALL_FIELDS_] = text_imputer.fit_transform(data[_ALL_FIELDS_])

    category_encoder = LabelEncoder()

    # Separate the features and the target variable
    X = combined_features.fit_transform(data)  # noqa: VNE001
    y = category_encoder.fit_transform(data[[_TARGET_COL_]])  # noqa: VNE001

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up the parameter dictionary for XGBoost
    # comment out various parameters for now and use the default for now
    # result seems to be the same
    params = {
        # "max_depth": 6,
        # "min_child_weight": 1,
        # "eta": 0.3,
        # "subsample": 1,
        # "colsample_bytree": 1,
        # "use_label_encoder": False,
        # "objective": "multi:softmax",
        # "num_class": len(category_encoder.classes_),
        # "eval_metric": "mlogloss",
        "verbosity": 2,
    }

    # Training the XGBoost classifier
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)

    # Making predictions and evaluating the model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    log.info(f"Accuracy: {accuracy}")

    # Filter le.classes_ to keep only those classes that were predicted
    all_categories = np.unique(np.concatenate((y_test, predictions)))
    all_labels = [label.strip()[:5] for label in category_encoder.classes_[all_categories]]
    report = classification_report(y_test, predictions, target_names=all_labels, output_dict=True)
    report = pd.DataFrame(report).T.to_dict()
    log.info("\n" + tabulate(report, headers="keys", tablefmt="plain", showindex=True, floatfmt=".2f"))

    joblib.dump(model, f"{model_prefix}_model.joblib")
    joblib.dump(combined_features, f"{model_prefix}_pipeline.joblib")
    joblib.dump(category_encoder, f"{model_prefix}_encoder.joblib")


def train_model_with_embedding(input_file: str):
    """very poor peformance, not maintained any more"""
    data = pd.read_csv(
        input_file,
        encoding="utf-8",
        dtype={field: str for field in _ALL_FIELDS_},  # force all fields to be string
    )

    # Separate the features and the target variable
    category_encoder = LabelEncoder()
    X = data.drop(["original_text", _TARGET_COL_], axis=1)  # noqa: VNE001
    y = category_encoder.fit_transform(data[[_TARGET_COL_]])  # noqa: VNE001

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # training parametrs
    params = {
        "max_depth": 6,
        "min_child_weight": 1,
        "eta": 0.3,
        "subsample": 1,
        "colsample_bytree": 1,
        "objective": "multi:softmax",
        "num_class": len(category_encoder.classes_),
        "eval_metric": "mlogloss",
        "verbosity": 2,
        "use_label_encoder": False,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)

    # Making predictions and evaluating the model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    log.info(f"Accuracy: {accuracy}")

    # Filter le.classes_ to keep only those classes that were predicted
    all_categories = np.unique(np.concatenate((y_test, predictions)))
    all_labels = [label.strip()[:5] for label in category_encoder.classes_[all_categories]]
    report = classification_report(y_test, predictions, target_names=all_labels)
    log.info("\n" + report)

    return model
