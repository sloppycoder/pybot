import pandas as pd

from bot.embedding import get_ada_embedding
from bot.utils import normalize_text


def test_get_embedding():
    chunk_size = 200

    full_df = pd.read_excel("data/test1.xlsx", sheet_name="输出（含人工分类结果）")

    tmp_df = full_df[["物料描述", "一级类目"]].applymap(normalize_text)
    tmp_df.rename({"物料描述": "original_text", "一级类目": "category"}, axis=1, inplace=True)

    original_texts = tmp_df["original_text"].tolist()
    chunks = [original_texts[i : i + chunk_size] for i in range(0, len(original_texts), chunk_size)]

    test_df = pd.DataFrame()
    count = 0
    for chunk in chunks:
        embeddings_list = [e.embedding for e in get_ada_embedding(chunk)]
        test_df = pd.concat([test_df, pd.DataFrame(embeddings_list)], axis=1)
        count += len(chunk)
        print(f"...{count}...\n")

    test_df = pd.concat([test_df, tmp_df], axis=1)
    test_df.to_csv("data/embedding1.csv", index=False)
