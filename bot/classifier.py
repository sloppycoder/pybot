import json
from datetime import datetime

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential


@retry(wait=wait_random_exponential(min=5, max=30), stop=stop_after_attempt(6))
def extract_features_with_openai(descriptions: list[str], debug: bool = False):
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
