import os

import openai


def test_openai_completion():
    prompt = "Hello world"
    model_name = os.environ.get("OPENAI_DEFAULT_MODEL", "gpt-3.5-turbo")
    completion = openai.ChatCompletion.create(model=model_name, messages=[{"role": "user", "content": prompt}])
    print(completion.choices[0].message.content)
