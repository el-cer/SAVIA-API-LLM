import pandas as pd
import re
import html
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # racine du projet
INPUT_CSV = os.path.join(PROJECT_ROOT, "knowledge", "raw", "free_all_articles.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "knowledge", "silver", "free_articles_cleaned.csv")

# --- Nettoyage du contenu
def clean_text(text):
    if pd.isna(text):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)  # retire les balises HTML
    text = re.sub(r"\s+", " ", text)      # espace propre
    return text.strip()

# --- Extraction mots-clés simples
def extract_keywords(text, top_n=5):
    words = re.findall(r"\b\w+\b", text.lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    most_common = Counter(words).most_common(top_n)
    return ", ".join([w for w, _ in most_common])

# --- Pipeline complet
def main():
    print(f"📥 Lecture de {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    df["content_clean"] = df["content"].apply(clean_text)
    df["keywords"] = df["content_clean"].apply(extract_keywords)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ {len(df)} articles nettoyés et sauvegardés dans : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
