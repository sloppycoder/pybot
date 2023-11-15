import csv
import os
import pickle
import pprint

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from bot.classifier import extract_features_with_openai
from bot.utils import chinese_tokenizer, chunked_iter, normalize_text, pickle_result

_TEXT_FEATURES_ = [
    "type",
    "function",
    "dimension",
    "brand",
    "model number",
    "component",
    "material",
]

_CATEGORY_ = "category"

_ALL_FIELDS_ = _TEXT_FEATURES_ + [_CATEGORY_]


def read_sample_set(csv_file: str) -> tuple[list[str], list[str]]:
    with open(csv_file, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file, delimiter=",")
        next(csv_reader)  # skip the header row

        rows = list(csv_reader)
        descriptions = [normalize_text(row[0]) for row in rows if len(row[0]) > 3]
        categories = [normalize_text(row[1]) for row in rows if len(row[0]) > 3]

        return descriptions, categories


def test_data_prep():
    description_category = {}

    with open("data/set1.csv", "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file, delimiter=",")
        next(csv_reader)  # skip the header row

        description_category = {}
        for row in csv_reader:
            if len(row[0]) > 3:
                description_category[normalize_text(row[0])] = normalize_text(row[1])

    with open("data/1k.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_ALL_FIELDS_)
        writer.writeheader()

        with open("data/result_1k.pickle", "rb") as file:
            results = pickle.load(file)
            print(f"total results: {len(results)}")
            for item in results:
                new_item = {key: item[key] for key in _ALL_FIELDS_ if key in item}
                desc = item["original_string"]
                try:
                    new_item["category"] = description_category[desc]
                except KeyError:
                    new_item["category"] = "unknown"
                writer.writerow(new_item)


def prep_features_df(input_file: str) -> pd.DataFrame:
    # Load your dataset
    df = pd.read_csv(input_file, encoding="utf-8")
    # data = data.head(15)

    # Check for missing values
    print("\n")
    print(df.isnull().sum())

    cat_imputer = SimpleImputer(strategy="constant", fill_value="unknown")
    for col in _ALL_FIELDS_:
        df[col] = cat_imputer.fit_transform(df[[col]])

    # Create a dataframe to hold our feature vectors
    features_df = pd.DataFrame()

    # For each text feature, we apply TF-IDF encoding with the Chinese tokenizer
    for text_col in _TEXT_FEATURES_:
        tfidf = TfidfVectorizer(tokenizer=chinese_tokenizer)
        tfidf_features = tfidf.fit_transform(df[text_col]).toarray()
        feature_labels = [f"{text_col}_{i}" for i in range(tfidf_features.shape[1])]
        features_tfidf_df = pd.DataFrame(tfidf_features, columns=feature_labels)
        features_df = pd.concat([features_df, features_tfidf_df], axis=1)

    # Encode the 'category' labels
    le = LabelEncoder()
    features_df[_CATEGORY_] = df[_CATEGORY_]
    features_df[_CATEGORY_] = le.fit_transform(features_df[_CATEGORY_])

    return features_df


def test_classify():
    data = prep_features_df("data/1k.csv")

    # Separate the features and the target variable
    X = data.drop(_CATEGORY_, axis=1)  # noqa: VNE001
    y = data[_CATEGORY_]  # noqa; VNE001

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the datasets into DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set up the parameter dictionary for XGBoost
    params = {
        "max_depth": 6,
        "min_child_weight": 1,
        "eta": 0.3,
        "subsample": 1,
        "colsample_bytree": 1,
        "objective": "multi:softmax",
        "num_class": 29,  # Number of classes in your target
        "eval_metric": "mlogloss",
    }

    # Train the model
    num_boost_round = 999
    model = xgb.train(
        params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, "Test")], early_stopping_rounds=10
    )

    # Predict the 'category' for the test set
    predictions = model.predict(dtest)
    # print(predictions)

    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)

    # If you want to see the predictions
    # print(predictions)


def test_extract_with_openai():
    total_results, results = 0, []
    descs, _ = read_sample_set("data/set1.csv")

    for chunk in chunked_iter(30, descs[:1000]):
        results += extract_features_with_openai(chunk, debug=os.environ.get("DEBUG", False))
        if len(results) > total_results:
            pickle_result(results)
            total_results = len(results)


def test_print_features():
    with open("result.pickle", "rb") as file:
        results = pickle.load(file)

        all_features = {}
        for item in results:
            if not isinstance(item, dict):
                # there're some items in the result we don't know what to do
                continue

            for key in item:
                if item[key]:
                    if key in all_features:
                        all_features[key].append(item[key])
                    else:
                        all_features[key] = [item[key]]

        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(all_features)

        print(f"total results: {len(results)}")
