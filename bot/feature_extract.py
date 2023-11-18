import json
import logging as log
from datetime import datetime
from typing import Any, Iterator

import openai
from openai.types.chat.chat_completion import ChatCompletion
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from bot import cache

_MODEL_MAP_ = {"35t": "gpt-3.5-turbo-1106", "4pre": "gpt-4-1106-preview"}


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


@retry(
    wait=wait_random_exponential(min=2, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type(InvalidResponse),
)
def invoke_openai_completion(parts: list[str], model_version: str) -> ChatCompletion:
    prompt_list = "\n".join(parts)

    start_t = datetime.now()
    log.debug(f"inovking API openai.chat.completions.create(...), input={prompt_list}")

    completion = openai.chat.completions.create(
        model=_MODEL_MAP_[model_version],
        messages=[
            {
                "role": "system",
                "content": """
                    I have a list of parts descriptions from some industrial enviroment. Each entry describes
                    a part used in a factory, typically consists simple description of the functionality
                    in Chinese and sometimes brand and model number too. I want to extract features from the strings.
                    Here are the featurs I'm interested in:
                        type, function, dimension, model_number,material.
                    If you see other features, just concatenate them into a single feature called "extra".
                    use "original_string" to save the original input. retain the original Chinese text for features
                    in output. Do not translate them into English.
                """,
            },
            {
                "role": "user",
                "content": f"""
                    I want to use json output format.
                    Please extract the features from the following list. treat each line as one input
                    {prompt_list}
                """,
            },
        ],
        response_format={"type": "json_object"},
    )

    if completion.choices[0].finish_reason != "stop":
        log.info(f"completion is not finished: reason={completion.choices[0].finish_reason}")
        raise InvalidResponse("completion is not finished")
    else:
        log.info(f"{len(parts)} inputs completed in {(datetime.now()-start_t).total_seconds()} seconds")
        log.info(completion.usage)

    reply = completion.choices[0].message.content

    if reply is None:
        raise InvalidResponse("Completion finished but reply is None")

    try:
        response = json.loads(reply)
        log.debug(json.dumps(response, indent=4, ensure_ascii=False))

        # the logic below counts number of responses and raise retry if
        # the output is not consistent with the input
        n_items = sum(1 for e in walk_response(response, parts))
        if n_items != len(parts):
            log.info(f"{len(parts)} intputs yieleded {n_items} outputs")
            # if len(parts) < 10 or abs(n_items - len(parts)) >= 2:
            # trigger retry only if the discrepenacy is large
            # TODO: check if should allow some mismatch in some cases
            raise InvalidResponse("number of inputs and outputs are not the same")

    except json.JSONDecodeError:
        log.warn("unable to parse output as json. got {reply}")
        raise InvalidResponse("unable to parse output as json")

    return completion


def extract_features_with_openai(items: list[str], model_version: str) -> list[dict]:
    features = {part: cache.find_extracted_features(part, model_version) for part in items}
    items_not_in_cache = [k for k, v in features.items() if v is None]

    if items_not_in_cache:
        try:
            completion = invoke_openai_completion(items_not_in_cache, model_version)
            response = json.loads(completion.choices[0].message.content)
            for item in walk_response(response, items_not_in_cache):
                if "original_string" in item:
                    features[item["original_string"]] = item
                    cache.save_extracted_feature(item["original_string"], model_version, item)
                else:
                    log.warn(f"original_text not found in response {item}")
        except RetryError:
            log.warn("failed after configured retries")

    # Use a list of keys to avoid RuntimeError for changing dictionary size during iteration
    for key in list(features.keys()):
        if features[key] is None:
            log.warn(f"unable to extract features for {key}")
            features.pop(key)

    return list(features.values())
