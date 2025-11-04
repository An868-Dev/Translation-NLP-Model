import pandas as pd
import re, string, warnings
from underthesea import word_tokenize
from collections import Counter

warnings.filterwarnings('ignore')

# ===================== LOAD & CLEAN DATA =====================
def load_and_preprocess(file_path: str, save_cleaned=True):
    df = pd.read_csv(file_path).dropna(subset=["en", "vi"])

    def clean_text(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r"\s+", " ", text.strip().lower())
        return text

    df["en"] = df["en"].apply(clean_text)
    df["vi"] = df["vi"].apply(clean_text)

    if save_cleaned:
        df.to_csv("file_song_ngu_cleaned.csv", index=False, encoding="utf-8")
    return df

# ===================== TOKENIZER =====================
def build_tokenizer_and_vocab(df):
    SRC, TGT = 'en', 'vi'
    token_transform, vocab_transform = {}, {}

    def en_tokenizer(text: str):
        return re.findall(r"\b\w+\b", text.lower())

    def vi_tokenizer(sentence: str):
        return word_tokenize(sentence)

    token_transform[SRC] = en_tokenizer
    token_transform[TGT] = vi_tokenizer

    # --- Build vocab manually ---
    def build_vocab(column, tokenizer):
        counter = Counter()
        for sent in column:
            counter.update(tokenizer(sent))
        vocab = ['<unk>', '<pad>', '<bos>', '<eos>'] + sorted(counter.keys())
        stoi = {word: i for i, word in enumerate(vocab)}
        itos = {i: word for word, i in stoi.items()}
        return {"vocab": vocab, "stoi": stoi, "itos": itos}

    vocab_transform[SRC] = build_vocab(df[SRC], token_transform[SRC])
    vocab_transform[TGT] = build_vocab(df[TGT], token_transform[TGT])

    return token_transform, vocab_transform

# ===================== TOKENIZE & SAVE =====================
def tokenize_and_save(df, token_transform, output_path="tokenized.csv"):
    df["en_tok"] = df["en"].apply(lambda x: " ".join(token_transform['en'](x)))
    df["vi_tok"] = df["vi"].apply(lambda x: " ".join(token_transform['vi'](x)))
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Tokenized file saved → {output_path}")

# ===================== RUN ALL =====================
if __name__ == "__main__":
    df = load_and_preprocess("file_song_ngu.csv")
    token_transform, vocab_transform = build_tokenizer_and_vocab(df)
    tokenize_and_save(df, token_transform)

    # Lưu vocab nếu muốn dùng cho training sau này
    import pickle
    with open("vocab_transform.pkl", "wb") as f:
        pickle.dump(vocab_transform, f)
    print("💾 Saved vocab → vocab_transform.pkl")
