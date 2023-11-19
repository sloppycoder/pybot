import jieba
import numpy as np

_FILLERS_ = [
    "NA",
    "na",
    "N/A",
    "n/a",
    "not available",
    "null",
    None,
]

_FEATURE_COLS_ = [
    "original_string",
    "type",
    "function",
    "dimension",
    "model_number",
    "material",
    "extra",
]

_TARGET_COL_ = "category"


# Function to tokenize Chinese text using jieba
def chinese_tokenizer(text) -> list[str]:
    return jieba.lcut(text)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    elif isinstance(text, float) and np.isnan(text):
        return ""
    elif isinstance(text, str):
        text = text.replace("\n", " ")
        text = text.replace("，", "")  # the wide comma
        text = text.replace(",", "")
        text = text.replace("'", "")
        text = text.replace('"', "")
        text = text.strip()
        text = text.replace("\xa0", " ")  # nospace break
        text = text.replace("   ", " ")
        text = text.replace("  ", " ")
        text = text.replace("  ", " ")
        text = text.replace("  ", " ")
        return text.strip()
    else:
        # actually it's not text, but...
        return text


def blank_filler(text: str):
    if text is None:
        return ""
    elif isinstance(text, str):
        return "" if text in _FILLERS_ else text
    elif isinstance(text, float) and np.isnan(text):
        return ""
    else:
        # may not be text actually...
        text
