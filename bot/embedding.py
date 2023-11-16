from typing import Any

import openai


def get_ada_embedding(texts: list[str]) -> list[Any]:
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-ada-002",
    )

    return response.data
