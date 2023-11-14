import csv
import itertools
import json
import os
import pickle
import pprint
from datetime import datetime

import openai


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    return text.strip()


def read_sample_set(csv_file: str) -> tuple[list[str], list[str]]:
    with open(csv_file, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file, delimiter=",")
        next(csv_reader)  # skip the header row

        rows = list(csv_reader)
        descriptions = [normalize_text(row[0]) for row in rows if len(row[0]) > 3]
        categories = [normalize_text(row[1]) for row in rows if len(row[0]) > 3]

        return descriptions, categories


def pickle_result(result: list[dict], file_name: str = "result.pickle") -> None:
    with open("result.pickle", "wb") as file:
        pickle.dump(result, file)


def group_result(result: list[dict[str, str]]) -> None:
    all_features: dict[str, list[str]] = {}
    for item in result:
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


def invoke_openai(descriptions: list[str], debug: bool = False):
    if debug:
        print(f"===DEBUG: input={','.join(descriptions)}")

    start_t = datetime.now()
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
                "content": """
                    extract the features from the following list of strings separtaed by comma.
                    I want to use json output format.
                    please assign all features to each of the strings in the following list:
                """
                + ",".join(descriptions),
            },
        ],
        response_format={"type": "json_object"},
    )

    if completion.choices[0].finish_reason != "stop":
        print("===WARN: completion is not finished")
        print(completion.choices[0].finish_reason)
        return []
    else:
        print(f"===INFO: {len(descriptions)} inputs completed in {(datetime.now()-start_t).total_seconds()} seconds")
        print(completion.usage)

    result = []
    reply = completion.choices[0].message.content
    if reply is not None:
        try:
            response = json.loads(reply)
            if debug:
                print(json.dumps(response, indent=4, ensure_ascii=False))

            # openai gpt4-1106-preview returns different structures
            #
            # type1:
            #
            #  {
            #       "products": [
            #           {"string":"sku1 desc1", "type": "determined by gpt"...},
            #           {"string":"sku1 desc2", "type": "determined by gpt"...},
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
            # the logic below handles these 2 types of outputs
            for key in response:
                if key in descriptions:  # this handles type 3
                    item = response[key]
                    result.append(item)
                else:
                    if isinstance(response[key], list):  # this handles type 1
                        result += response[key]
                    else:  # this handles type 2
                        result.append(response[key])
        except json.JSONDecodeError:
            print("unable to parse output as json ===")
            print(reply)
    else:
        print("reply is None")
        print(completion.choices[0].message)

    if len(result) != len(descriptions):
        print(f"===WARN: {len(descriptions)} intputs yieleded {len(result)} outputs")

    return result


def test_extract_with_openai():
    result = []
    chunk_size = 30
    total_results = 0

    descs, _ = read_sample_set("data/set1.csv")

    my_iteractor = iter(descs[:1000])
    while True:
        chunk = list(itertools.islice(my_iteractor, chunk_size))
        if not chunk:
            break

        result += invoke_openai(chunk, debug=os.environ.get("DEBUG", False))
        if len(result) > total_results:
            pickle_result(result)
            total_results = len(result)

    group_result(result)


def test_print_features():
    with open("result.pickle", "rb") as file:
        results = pickle.load(file)
        group_result(results)
        print(f"total results: {len(results)}")
