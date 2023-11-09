import json
import os

import openai


def pretty_print_json(json_string):
    parsed = json.loads(json_string)
    pretty = json.dumps(parsed, indent=4)
    print(pretty)


def test_openai_completion():
    prompt = "Hello world"
    model_name = os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo")
    completion = openai.ChatCompletion.create(model=model_name, messages=[{"role": "user", "content": prompt}])
    print(completion.choices[0].message.content)


def test_openai_codegen():
    prompt = """
    please create a python project that:
       1. uses poetry as package manager, python 3.11 and pytest
       2. the project implements a REST API using FastAPI framework
       3. The REST API handles entity Customer, which has the following fields:
            name, date_of_birth, mobile, email
       4. API should handle different HTTP verb and return proper HTTP status codes
       5. the project should include pytest test cases for all APIs
    """
    completion = openai.chat.completions.create(
        # model="gpt-3.5-turbo-1106",
        model="gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "generate code in python use '===code===' tag in separate generated code from other elements. use json output format",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format={"type": "text"},
    )
    reply = completion.choices[0].message.content
    print(reply)
    # pretty_print_json(reply)
