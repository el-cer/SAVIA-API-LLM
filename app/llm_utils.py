import os
from llama_cpp import Llama
from dotenv import load_dotenv
import hdbscan, joblib, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

load_dotenv()

# ============================================================
# 🔧 Chargement des modèles
# ============================================================

MODEL_NAME = "dangvantuan/sentence-camembert-large"
MAX_TOKENS = 510

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
embedder = SentenceTransformer(MODEL_NAME)
embedder.tokenizer = tokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HDBSCAN_PATH = os.path.join(PROJECT_ROOT, "knowledge", "gold", "hdbscan_model.joblib")
clusterer = joblib.load(HDBSCAN_PATH)

# ============================================================
# 🔹 Embeddings utilisateur
# ============================================================

def truncate_text(text, max_tokens=MAX_TOKENS):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
        text = tokenizer.convert_tokens_to_string(tokens)
    return text

def compute_embedding(text: str):
    return embedder.encode([truncate_text(text)], normalize_embeddings=True)[0]

# ============================================================
# 🔹 Génération de réponse — local (streaming)
# ============================================================

def generate_response_stream(model: Llama, prompt: str, max_tokens: int = 256):
    """
    Génère un flux de texte depuis le modèle local (llama.cpp)
    """
    try:
        # 🧠 modèle local en mode streaming
        stream = model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            stream=True,
            temperature=0.7,
            top_p=1.0,
        )

        for chunk in stream:
            if "choices" in chunk:
                delta = chunk["choices"][0].get("text", "")
                if delta:
                    yield delta
        yield "\n"

    except Exception as e:
        yield f"\n[ERREUR STREAM LOCAL] {e}"

# ============================================================
# 🔹 API Mistral — version streaming persistante
# ============================================================

def request_model_api(
    prompt: str,
    context: str = "",
    model_name: str = "mistral-medium-latest",
):
    """
    Appel à l'API Mistral (sans stream natif) avec yield progressif.
    Compatible mistralai >= 1.3.0.
    """
    import os
    import time
    from mistralai import Mistral

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("❌ MISTRAL_API_KEY manquante dans l'environnement")

    client = Mistral(api_key=api_key)

    # 🔹 Contexte système
    system_prompt = (
        context.strip()
        or "Tu es un agent SAV Free. Tes réponses doivent être claires, concises et empathiques."
    )

    # 🔹 Messages structurés
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt.strip()},
    ]

    # 🔁 Générateur de texte progressif
    def generator():
        try:
            print("🚀 [MISTRAL] Appel API complet...")

            # Appel complet (non-stream)
            response = client.chat.complete(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
                top_p=1.0,
            )

            # ✅ Correction ici : accéder à message.content (pas ["content"])
            result = response.choices[0].message.content

            print("✅ [MISTRAL] Réponse reçue, envoi progressif...")

            # 🪶 Simulation de flux (mot par mot)
            for word in result.split():
                yield word + " "
                time.sleep(0.03)  # effet "stream" fluide

            yield "\n"

        except Exception as e:
            print(f"❌ [ERREUR API MISTRAL] {e}")
            yield f"⚠️ Erreur lors de l’appel à l’API Mistral : {e}"

    return generator()



# ============================================================
# 🔹 Top-K contextes RAG
# ============================================================

def get_top_k_contexts(user_embedding: np.ndarray, df_gold, k: int = 3):
    predicted_label, _ = hdbscan.approximate_predict(clusterer, [user_embedding])
    cluster_id = predicted_label[0]

    if cluster_id == -1:
        gold_embeddings = np.vstack(df_gold["embedding"].values)
        sims = cosine_similarity([user_embedding], gold_embeddings)[0]
        top_indices = sims.argsort()[-k:][::-1]
        return df_gold.iloc[top_indices]["content_clean"].tolist()

    cluster_df = df_gold[df_gold["cluster"] == cluster_id]
    if cluster_df.empty:
        gold_embeddings = np.vstack(df_gold["embedding"].values)
        sims = cosine_similarity([user_embedding], gold_embeddings)[0]
        top_indices = sims.argsort()[-k:][::-1]
        return df_gold.iloc[top_indices]["content_clean"].tolist()

    cluster_embeddings = np.vstack(cluster_df["embedding"].values)
    sims = cosine_similarity([user_embedding], cluster_embeddings)[0]
    top_indices = sims.argsort()[-k:][::-1]
    return cluster_df.iloc[top_indices]["content_clean"].tolist()

# ============================================================
# 🔹 Classification simple
# ============================================================

def classify_text(prompt: str, df_gold):
    prompt_embedding = compute_embedding(prompt)
    gold_embeddings = np.vstack(df_gold["embedding"].values)
    sims = cosine_similarity([prompt_embedding], gold_embeddings)[0]
    best_idx = np.argmax(sims)
    return df_gold.iloc[best_idx]["label"], float(sims[best_idx])
