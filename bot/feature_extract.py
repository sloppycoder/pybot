import hashlib
import json
import os
from datetime import datetime
from typing import Any, Iterator

import openai
import pandas as pd
import redis
from openai.types.chat.chat_completion import ChatCompletion
from pydantic.tools import parse_obj_as
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from .classification import _UNKNOWN_

debug = os.environ.get("DEBUG", False)

_CACHE_VALIDITY_ = 3600 * 72 * 30

cache = redis.Redis(host="localhost", port=6379, db=0)

if os.environ.get("CLEAR_CACHE", False):
    keys = cache.keys("openai*")
    cache.delete(*keys)  # type: ignore


class InvalidResponse(Exception):
    pass


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
        # for some reason openai returns both "model_number" and "model number" for some entires
        if "model_number" in item:
            model_num = item.pop("model_number")
            if "model number" not in item:
                item["model number"] = model_num
            else:
                item["model number"] += f" {model_num}"

        result_df = pd.concat([result_df, pd.DataFrame([item])], ignore_index=True)

    categories = []
    for _, row in result_df.iterrows():
        try:
            categories.append(input_df[input_df["description"] == row["original_string"]]["category"].values[0])
        except IndexError:
            categories.append(_UNKNOWN_)
    result_df["category"] = categories

    return result_df
