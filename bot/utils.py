import jieba


# Function to tokenize Chinese text using jieba
def chinese_tokenizer(text):
    return jieba.lcut(text)


def normalize_text(text: str) -> str:
    if isinstance(text, str):
        text = text.replace("'", "")
        text = text.replace('"', "")
        text = text.replace("\n", " ")
        text = text.replace("  ", " ")
        return text.strip()
    else:
        # actually it's not text, but...
        return text
