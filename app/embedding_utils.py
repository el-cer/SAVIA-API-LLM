# embedding_utils.py

from sentence_transformers import SentenceTransformer
import numpy as np

# 🔹 Modèle français basé sur CamemBERT
EMBEDDING_MODEL = "dangvantuan/sentence-camembert-large"
embedder = SentenceTransformer(EMBEDDING_MODEL)

def embed_text(text: str) -> np.ndarray:
    """
    Encode un texte en embedding numpy (CamemBERT).
    """
    return embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calcule la similarité cosinus entre deux vecteurs.
    """
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
