import csv
import pickle

import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    return text.strip()


def data_prep():
    skipped = []
    description_category = {}
    with open("data/set1.csv", "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file, delimiter=",")
        next(csv_reader)  # skip the header row

        description_category = {}
        for row in csv_reader:
            if len(row[0]) > 3:
                description_category[normalize_text(row[0])] = normalize_text(row[1])

    keys_to_preserve = [
        "type",
        "function",
        "dimension",
        "brand",
        "model number",
        "component",
        "material",
        "category",
    ]
    default = "unknown"

    with open("data/1k.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys_to_preserve)
        writer.writeheader()

        with open("data/result_1k.pickle", "rb") as file:
            results = pickle.load(file)
            print(f"total results: {len(results)}")
            for item in results:
                desc = item["original_string"]
                try:
                    item["category"] = description_category[desc]
                except KeyError:
                    skipped.append(desc)

                new_item = {
                    key: item[key] if key in item and item[key] is not None else default for key in keys_to_preserve
                }
                print(new_item)
                writer.writerow(new_item)

    print(f"=== skipped {len(skipped)} items ===")
    print(skipped)


def test_data_prep():
    data_prep()


def test_classify():
    # Load your dataset
    data = pd.read_csv("data/1k.csv")

    # Check for missing values
    print(data.isnull().sum())

    # # If there are missing values, you can either impute them or drop the rows with missing values. For example, to impute missing values with the mean, you can use the following code:
    # imputer = SimpleImputer(missing_values="NaN", strategy="mean")
    # data = imputer.fit_transform(data)

    # Separate the features and the target variable
    X = data.drop("category", axis=1)
    y = data["category"]

    # Convert categorical features to numerical features
    encoder = OneHotEncoder()
    X = encoder.fit_transform(X)

    # Split the data into training and test sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    # Create and train an XGBoost classifier
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model's performance
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
