import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# --- Base directory = racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Paramètres
MODEL_NAME = "dangvantuan/sentence-camembert-large"
MAX_TOKENS = 510  # CamemBERT accepte max 512 tokens
INPUT_CSV = os.path.join(PROJECT_ROOT, "knowledge", "silver", "free_articles_cleaned.csv")
OUTPUT_PARQUET = os.path.join(PROJECT_ROOT, "knowledge", "gold", "articles_with_embeddings.parquet")

# --- Chargement du modèle et tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = SentenceTransformer(MODEL_NAME)
model.tokenizer = tokenizer

# --- Chargement des données nettoyées
df = pd.read_csv(INPUT_CSV)
assert "content_clean" in df.columns, "❌ La colonne 'content_clean' est manquante."

def truncate_text(text: str) -> str:
    """
    Tronque un texte à MAX_TOKENS tokens pour éviter les erreurs du modèle.
    """
    tokens = tokenizer.tokenize(text)
    if len(tokens) > MAX_TOKENS:
        tokens = tokens[:MAX_TOKENS]
        text = tokenizer.convert_tokens_to_string(tokens)
    return text

# --- Embedding avec protection contre dépassement de longueur
tqdm.pandas(desc="🔍 Embedding avec CamemBERT (tronqué à 510 tokens)")
df["embedding"] = df["content_clean"].progress_apply(
    lambda x: model.encode(truncate_text(x), normalize_embeddings=True).tolist()
)

# --- Sauvegarde
os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)
df.to_parquet(OUTPUT_PARQUET, index=False)
print(f"✅ Embeddings générés et sauvegardés dans : {OUTPUT_PARQUET}")
