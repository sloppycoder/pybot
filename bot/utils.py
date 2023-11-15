import itertools
import pickle
from typing import Iterator

import jieba


# Function to tokenize Chinese text using jieba
def chinese_tokenizer(text):
    return jieba.lcut(text)


def normalize_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = text.replace("  ", " ")
    return text.strip()


def pickle_result(result: list[dict], file_name: str = "result.pickle") -> None:
    with open(file_name, "wb") as file:
        pickle.dump(result, file)


def chunked_iter(chunk_size: int, my_list: list) -> Iterator[list]:
    my_iteractor = iter(my_list)
    while True:
        chunk = list(itertools.islice(my_iteractor, chunk_size))
        if not chunk:
            return

        yield chunk
