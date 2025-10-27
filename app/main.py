from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import pandas as pd
import os

from app.model_loader import load_model
from app.llm_utils import (
    generate_response_stream,
    request_model_api,
    get_top_k_contexts,
    classify_text,
)
from app.embedding_utils import embed_text

# ============================================================
#  INIT
# ============================================================

app = FastAPI()

# Autoriser ton frontend Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://192.168.1.144:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Charger modèle local
model = load_model()

# Charger ton fichier gold
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD_PATH = os.path.join(PROJECT_ROOT, "knowledge", "gold", "articles_with_embeddings.parquet")
df_gold = pd.read_parquet(GOLD_PATH)

assert "embedding" in df_gold.columns
assert "content_clean" in df_gold.columns

# Contexte par défaut
DEFAULT_CONTEXT = (
    "Tu es un assistant SAV Free. "
    "Tu aides les clients à résoudre leurs problèmes avec leur Freebox (voyant rouge, Wi-Fi, erreurs de configuration...). "
    "Réponds en français, sans politesse inutile, en étapes numérotées claires et concises."
)

default_classifications = ["problème avéré", "problème non avéré"]
domaines = ["mobile", "fixe", "facture"]
sous_domaines = ["réseau", "wifi", "box","appel voix","sécurité"]
                 

CLASSIFICATION_CONTEXT = (
    "Tu es un assistant SAV Free. "
    "Ton rôle est de classifier un tweet en fonction de son contenu. "
    "Tu dois retourner un JSON sous la forme : "
    "{'label': label, 'domaine': domaine, 'sous_domaine': sous_domaine, 'text': prompt}. "
    f"Voici les possibilités pour 'label' (classification binaire) : {default_classification}. "
    f"Voici les possibilités pour 'domaine' : {domaine}. "
    f"Voici les possibilités pour 'sous_domaine' : {sous_domaine}. "
    "Analyse bien le texte et choisis la catégorie la plus pertinente."
)


# ============================================================
#  Schémas de requêtes
# ============================================================

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 256
    context: str = DEFAULT_CONTEXT
    model_selected: str

class ClassifyRequest(BaseModel):
    prompt: str
    model: str
    context: str

# ============================================================
#  Chat principal
# ============================================================

@app.post("/chat_sav")
def chat_sav(req: PromptRequest):
    start = time.time()

    # 🧠 Cas 1 — LLM local (mistral-7b-instruct)
    if req.model_selected == "Mistral-7B-Instruct":
        user_embedding = embed_text(req.prompt)
        top_contexts = get_top_k_contexts(user_embedding, df_gold, k=3)
        retrieved_context = "\n\n".join(top_contexts)

        full_prompt = (
            f"{req.context.strip() or DEFAULT_CONTEXT}\n\n"
            f"{retrieved_context}\n\n"
            f"Client : {req.prompt}\n"
            f"Assistant (réponds en étapes numérotées, de manière empathique et concise) :"
        )

        stream = generate_response_stream(model, full_prompt, req.max_tokens)
        return StreamingResponse(stream, media_type="text/plain")

    # ☁️ Cas 2 — API Mistral
    elif req.model_selected == "Mistral-medium":
        stream = request_model_api(
            prompt=req.prompt,
            context=req.context or DEFAULT_CONTEXT,
            model_name="mistral-medium-latest",
        )
        return StreamingResponse(stream, media_type="text/plain")

    return {"response": "❌ Modèle non reconnu."}

# ============================================================
# Classification
# ============================================================


@app.post("/classify")
def classify(req: ClassifyRequest):
    label, domaine, sous_domaine = classify_text(
        req.prompt,
        req.model,
        default_classification,
        domaines,
        sous_domaines,
        req.context.strip() or CLASSIFICATION_CONTEXT
    )
    return {"label": label, "domaine": domaine, "sous_domaine": sous_domaine, "text": req.prompt}


@app.get("/health")
def health_check():
    return {"status": "ok"}
