import argparse
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', ' ', text)         # remove HTML
    text = re.sub(r'http\S+|www\S+', ' ', text)  # remove URLs
    text = re.sub(r'[^A-Za-z\s]', ' ', text)  # keep letters
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    stops = set(stopwords.words('english'))
    lem = WordNetLemmatizer()
    tokens = [lem.lemmatize(t) for t in tokens if t not in stops and len(t) > 2]
    return ' '.join(tokens)

def build_dataframe(fake_path: str, real_path: str) -> pd.DataFrame:
    df_fake = pd.read_csv(fake_path)
    df_real = pd.read_csv(real_path)

    # Expect columns like 'text' or 'title','text'
    def get_text(df):
        if 'text' in df.columns:
            return df['text']
        elif 'content' in df.columns:
            return df['content']
        else:
            # combine title + text if available
            cols = [c for c in df.columns if c.lower() in ['title','subject','date','text','content']]
            return df[cols].astype(str).agg(' '.join, axis=1)

    fake_text = get_text(df_fake)
    real_text = get_text(df_real)

    df_fake_c = pd.DataFrame({'text': fake_text, 'label': 0})
    df_real_c = pd.DataFrame({'text': real_text, 'label': 1})
    df = pd.concat([df_fake_c, df_real_c], axis=0, ignore_index=True).dropna()
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.strip().str.len() > 0].reset_index(drop=True)
    return df

def split_and_save(df: pd.DataFrame, out_dir: str, val_size: float=0.1, test_size: float=0.1, seed: int=42):
    train_df, temp_df = train_test_split(df, test_size=val_size+test_size, random_state=seed, stratify=df['label'])
    rel_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(temp_df, test_size=rel_test_size, random_state=seed, stratify=temp_df['label'])
    train_df.to_csv(f"{out_dir}/train.csv", index=False)
    val_df.to_csv(f"{out_dir}/val.csv", index=False)
    test_df.to_csv(f"{out_dir}/test.csv", index=False)
    print(f"Saved: {out_dir}/train.csv ({len(train_df)}), {out_dir}/val.csv ({len(val_df)}), {out_dir}/test.csv ({len(test_df)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fake_path', required=True, help='Path to Fake.csv')
    ap.add_argument('--real_path', required=True, help='Path to True.csv')
    ap.add_argument('--out_dir', default='data')
    ap.add_argument('--val_size', type=float, default=0.1)
    ap.add_argument('--test_size', type=float, default=0.1)
    args = ap.parse_args()

    df = build_dataframe(args.fake_path, args.real_path)
    split_and_save(df, args.out_dir, val_size=args.val_size, test_size=args.test_size)

if __name__ == '__main__':
    main()
