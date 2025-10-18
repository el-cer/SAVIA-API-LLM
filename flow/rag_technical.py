import os
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# Chargement du modèle
def load_model_embedder():
    model_name = "dangvantuan/sentence-camembert-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = SentenceTransformer(model_name)
    model.tokenizer = tokenizer
    return model

# Détection du chemin dynamique
def load_technical_embeddings():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    emb_path = os.path.join(base_dir, "knowledge", "gold", "tech_keywords.pt")
    data = torch.load(emb_path)
    return data["keywords"], data["embeddings"]

# Fonction de comparaison
def word_domain(technical_embedded, sentence, model, keywords, threshold=0.45):
    sent_emb = model.encode(sentence, convert_to_tensor=True, normalize_embeddings=True)
    sim_scores = cosine_similarity(technical_embedded.cpu(), sent_emb.cpu().unsqueeze(0))

    # Affichage filtré
    for mot, score in sorted(zip(keywords, sim_scores.flatten()), key=lambda x: x[1], reverse=True):
        tag = "✅" if score > threshold else "❌"
        print(f"{tag} {mot:20s} → Similarité : {score:.4f}")
    
    if sim_scores.max() > threshold:
        print("\n✅ Phrase considérée comme TECHNIQUE.")
    else:
        print("\n❌ Phrase considérée comme NON TECHNIQUE.")

# Exécution
if __name__ == "__main__":
    model = load_model_embedder()
    keywords, embeddings = load_technical_embeddings()

    test_phrase = "La freebox à un très bon wifi"
    word_domain(embeddings, test_phrase, model, keywords)
