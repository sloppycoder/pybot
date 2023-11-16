import hashlib
import json
import os
from datetime import datetime
from typing import Any, Iterator

import openai
import pandas as pd
import redis
import xgboost as xgb
from openai.types.chat.chat_completion import ChatCompletion
from pydantic.tools import parse_obj_as
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from bot.utils import chinese_tokenizer

debug = os.environ.get("DEBUG", False)

_CACHE_VALIDITY_ = 3600 * 72

cache = redis.Redis(host="localhost", port=6379, db=0)

if os.environ.get("CLEAR_CACHE", False):
    keys = cache.keys("openai*")
    cache.delete(*keys)  # type: ignore


class InvalidResponse(Exception):
    pass


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


def prep_features_df(input_file: str) -> pd.DataFrame:
    # Load your dataset
    df = pd.read_csv(
        input_file,
        encoding="utf-8",
        dtype={field: str for field in _ALL_FIELDS_},  # force all fields to be string
    )
    # data = data.head(15)

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


def train_classification_model(input_file: str):
    data = prep_features_df(input_file)

    # Separate the features and the target variable
    X = data.drop(_CATEGORY_, axis=1)  # noqa: VNE001
    y = data[_CATEGORY_]  # noqa; VNE001

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
        "num_class": 31,  # there're 31 categories, including the unknown category
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

    return model


def walk_response(response: Any, parts: list[str]) -> Iterator[dict]:
    # openai gpt4-1106-preview returns different structures
    #
    # type1:
    #
    #  {
    #       "products": [
    #           {"original_string":"sku1 desc1", "type": "determined by gpt"...},
    #           {"original_string":"sku1 desc2", "type": "determined by gpt"...},
    #       ]
    # }
    # key product sometimes can be features, items, etc
    #
    # type2:
    #
    #  {
    #      "1": { "type": "determined by gpt"...},
    #      "2": { "type": "determined by gpt"...},
    # },
    #
    # type3:
    # {
    #       "sku1 desc1" : {"type": "determined by gpt"...},
    #       "sku1 desc2" : {"type": "determined by gpt"...},
    # }
    #
    # type4:
    #   when input is a single item the output is only one single dict
    #  {"original_string":"sku1 desc1", "type": "determined by gpt"...},
    #
    if "original_string" in response:
        # type4: single dict
        yield response
    else:
        for key in response:
            if key in parts:
                # this handles type 3
                yield response[key]
            else:
                if isinstance(response[key], list):
                    # this handles type 2
                    for item in response[key]:
                        yield item
                else:
                    # this handles type 1
                    yield response[key]


def cache_key(obj: Any) -> str:
    """use sha1 hash a json string as the key"""
    sha1_hash = hashlib.sha1()
    sha1_hash.update(json.dumps(obj).encode("utf-8"))
    hex_digest = sha1_hash.hexdigest()
    return f"openai-{hex_digest}"


@retry(
    wait=wait_random_exponential(min=2, max=60),
    stop=stop_after_attempt(4),
    retry=retry_if_exception_type(InvalidResponse),
)
def invoke_openai_completion(parts: list[str]) -> ChatCompletion:
    prompt_list = ",".join(parts)

    start_t = datetime.now()
    if debug:
        print(f"===DEBUG: inovking API openai.chat.completions.create(...), input={prompt_list}")

    completion = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": """
                    i want you to extract features from a series of stirngs, each string typically consists of
                    a SKU and a brief description with Chinese characters. the features I'm interested in are
                    "type", "function", "dimension", "brand", "model number", "component", "material".
                    use "original_string" to save the original input.
                    retain the original Chinese text for features in output. do not translate them into English
                """,
            },
            {
                "role": "user",
                "content": f"""
                    extract the features from the following list of strings separtaed by comma.
                    I want to use json output format.
                    please assign all features to each of the strings in the following list:
                    {prompt_list}
                """,
            },
        ],
        response_format={"type": "json_object"},
    )

    if completion.choices[0].finish_reason != "stop":
        print("===WARN: completion is not finished")
        print(completion.choices[0].finish_reason)
        raise InvalidResponse("completion is not finished")
    else:
        print(f"===INFO: {len(parts)} inputs completed in {(datetime.now()-start_t).total_seconds()} seconds")
        print(completion.usage)

    reply = completion.choices[0].message.content

    if reply is None:
        raise InvalidResponse("Completion finished but reply is None")

    try:
        response = json.loads(reply)
        if debug:
            print(json.dumps(response, indent=4, ensure_ascii=False))

        # the logic below counts number of responses and raise retry if
        # the output is not consistent with the input
        n_items = sum(1 for e in walk_response(response, parts))
        if n_items != len(parts):
            print(f"===WARN: {len(parts)} intputs yieleded {n_items} outputs")
            if len(parts) < 10 or abs(n_items - len(parts)) >= 2:
                # trigger retry only if the discrepenacy is large
                raise InvalidResponse("number of inputs and outputs are not the same")

    except json.JSONDecodeError:
        print("===WARN: unable to parse output as json")
        print(reply)
        raise InvalidResponse("unable to parse output as json")

    return completion


def get_openai_response(parts: list[str]) -> Any:
    key = cache_key(",".join(parts))

    cached_result = cache.get(key)
    if cached_result is not None:
        completion = parse_obj_as(ChatCompletion, json.loads(cached_result))  # type: ignore
    else:
        completion = invoke_openai_completion(parts)
        # store the api response to cache
        cache.set(key, json.dumps(completion.model_dump()), ex=_CACHE_VALIDITY_)
        # store the index of part in cache also
        with cache.pipeline() as pipeline:
            for part in parts:
                pipeline.hset("openai-parts-index", part, key)

    reply = completion.choices[0].message.content
    return json.loads(reply)


def extract_features_with_openai(input_df: pd.DataFrame) -> pd.DataFrame:
    parts = input_df["description"].tolist()
    response = get_openai_response(parts)

    result_df = pd.DataFrame()
    for item in walk_response(response, parts):
        result_df = pd.concat([result_df, pd.DataFrame([item])], ignore_index=True)

    categories = []
    for _, row in result_df.iterrows():
        try:
            categories.append(input_df[input_df["description"] == row["original_string"]]["category"].values[0])
        except IndexError:
            categories.append(_UNKNOWN_CATEGORY_)
    result_df["category"] = categories

    return result_df
