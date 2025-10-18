import pandas as pd
import re
import html
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import os

# --- Optionnel : stopwords français
try:
    from nltk.corpus import stopwords
    FRENCH_STOPWORDS = set(stopwords.words("french"))
except LookupError:
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    FRENCH_STOPWORDS = set(stopwords.words("french"))

ALL_STOPWORDS = ENGLISH_STOP_WORDS.union(FRENCH_STOPWORDS)

# --- Fichiers et chemins
INPUT_CSV = os.path.join("knowledge", "raw", "free_all_articles.csv")
OUTPUT_CSV = os.path.join("knowledge", "silver", "free_articles_cleaned.csv")

# --- Nettoyage du texte HTML brut
def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = html.unescape(text)                        # Unescape HTML
    text = re.sub(r"<[^>]+>", " ", text)              # Supprime balises HTML
    text = re.sub(r"[^\w\sÀ-ÿ]", " ", text)           # Garde lettres + accents
    text = re.sub(r"\s+", " ", text)                  # Supprime espaces multiples
    return text.strip()

# --- Extraction de mots-clés simples
def extract_keywords(text: str, top_n: int = 5) -> str:
    words = re.findall(r"\b\w+\b", text.lower())
    words = [w for w in words if w not in ALL_STOPWORDS and len(w) > 2]
    most_common = Counter(words).most_common(top_n)
    return ", ".join([w for w, _ in most_common])

# --- Pipeline principal
def main():
    print(f"📥 Lecture : {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    df["content_clean"] = df["content"].apply(clean_text)
    df["keywords"] = df["content_clean"].apply(extract_keywords)

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ {len(df)} articles nettoyés et sauvegardés dans : {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
