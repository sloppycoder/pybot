import jieba

_FILLERS_ = [
    "N/A",
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
    if isinstance(text, str):
        text = text.replace("，", "")
        text = text.replace(",", "")
        text = text.replace("'", "")
        text = text.replace('"', "")
        text = text.replace("\n", " ")
        text = text.replace("\xa0", " ")
        text = text.replace("   ", " ")
        text = text.replace("  ", " ")
        text = text.replace("  ", " ")
        text = text.replace("  ", " ")
        return text.strip()
    else:
        # actually it's not text, but...
        return text


def remove_fillers(items):
    for item in items:
        for col in _FEATURE_COLS_:
            if col not in item or item[col] in _FILLERS_:
                item[col] = ""
    return items
