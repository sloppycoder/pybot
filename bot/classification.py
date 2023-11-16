import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from bot.utils import chinese_tokenizer

debug = os.environ.get("DEBUG", False)


_TEXT_FEATURES_ = [
    "original_string",
    "type",
    "function",
    "dimension",
    "brand",
    "model number",
    "component",
    "material",
]

_CATEGORY_ = "category"
_UNKNOWN_CATEGORY_ = "待定"
_UNKNOWN_FEATURE_ = "不详"

_ALL_FIELDS_ = _TEXT_FEATURES_ + [_CATEGORY_]

category_encoder = LabelEncoder()

# category_encoder = OneHotEncoder(sparse_output=False)


class PartsClassifier:
    def __init__(self, model_path: str):
        self.model = xgb.Booster({"nthread": 4})
        self.model.load_model(model_path)

    def guess(self, part: str):
        pass
        # part_vec = part  # convert part from string into some vector
        # pred = self.model.predict(part_vec)
        # return pred


def prep_features_df(input_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        input_file,
        encoding="utf-8",
        dtype={field: str for field in _ALL_FIELDS_},  # force all fields to be string
    )

    # drop columsn we're not interested in
    df = df.drop([col for col in df.columns if col not in _ALL_FIELDS_], axis=1)

    # Check for missing values
    print("\n")
    print(df.isnull().sum())

    feature_imputer = SimpleImputer(strategy="constant", fill_value=_UNKNOWN_FEATURE_)
    for col in _TEXT_FEATURES_:
        df[col] = feature_imputer.fit_transform(df[[col]])

    category_imputer = SimpleImputer(strategy="constant", fill_value=_UNKNOWN_CATEGORY_)
    df[_CATEGORY_] = category_imputer.fit_transform(df[[_CATEGORY_]])

    # Create a dataframe to hold our feature vectors
    features_df = pd.DataFrame()

    # For each text feature, we apply TF-IDF encoding with the Chinese tokenizer
    tfidf = TfidfVectorizer(tokenizer=chinese_tokenizer)
    for text_col in _TEXT_FEATURES_:
        tfidf_features = tfidf.fit_transform(df[text_col]).toarray()
        feature_labels = [f"{text_col}_{i}" for i in range(tfidf_features.shape[1])]
        features_tfidf_df = pd.DataFrame(tfidf_features, columns=feature_labels)
        features_df = pd.concat([features_df, features_tfidf_df], axis=1)

    # # Encode the 'category' labels
    features_df[_CATEGORY_] = df[_CATEGORY_]

    return features_df


def train_model_with_embedding(input_file: str):
    data = pd.read_csv(
        input_file,
        encoding="utf-8",
        dtype={field: str for field in _ALL_FIELDS_},  # force all fields to be string
    )

    # Separate the features and the target variable
    X = data.drop(["original_text", _CATEGORY_], axis=1)  # noqa: VNE001
    y = category_encoder.fit_transform(data[[_CATEGORY_]])  # noqa: VNE001

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # training parametrs
    params = {
        "max_depth": 6,
        "min_child_weight": 1,
        "eta": 0.3,
        "subsample": 1,
        "colsample_bytree": 1,
        "objective": "multi:softmax",
        "num_class": 30,  # there're 31 categories, including the unknown category
        "eval_metric": "mlogloss",
        "verbosity": 2,
        "use_label_encoder": False,
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10)

    # Making predictions and evaluating the model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    # Filter le.classes_ to keep only those classes that were predicted
    all_categories = np.unique(np.concatenate((y_test, predictions)))
    all_labels = [label.strip()[:5] for label in category_encoder.classes_[all_categories]]
    report = classification_report(y_test, predictions, target_names=all_labels)
    print(report)

    return model


def train_model_with_features(input_file: str):
    data = prep_features_df(input_file)

    # Separate the features and the target variable
    X = data.drop(_CATEGORY_, axis=1)  # noqa: VNE001
    y = category_encoder.fit_transform(data[[_CATEGORY_]])  # noqa: VNE001

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set up the parameter dictionary for XGBoost
    params = {
        "max_depth": 6,
        "min_child_weight": 1,
        "eta": 0.3,
        "subsample": 1,
        "colsample_bytree": 1,
        "objective": "multi:softmax",
        "num_class": 30,  # there're 31 categories, including the unknown category
        "eval_metric": "mlogloss",
        "verbosity": 2,
        "use_label_encoder": False,
    }

    # Training the XGBoost classifier
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Making predictions and evaluating the model
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")

    # Filter le.classes_ to keep only those classes that were predicted
    all_categories = np.unique(np.concatenate((y_test, predictions)))
    all_labels = [label.strip()[:5] for label in category_encoder.classes_[all_categories]]
    report = classification_report(y_test, predictions, target_names=all_labels)
    print(report)

    return model
