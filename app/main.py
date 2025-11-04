from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
import pandas as pd
import os
import json
import re
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
    allow_origins=["*"],
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
    "Ton rôle est de classifier un tweet ou message client selon son contenu. "
    "Tu dois renvoyer un JSON **valide** au format suivant : "
    "{'label': label, 'domaine': domaine, 'sous_domaine': sous_domaine, 'text': prompt}. "

    "Critères :\n"
    "- Si le message décrit un dysfonctionnement, une coupure, un bug, une lenteur, ou une plainte explicite → 'label': 'problème avéré'.\n"
    "- Si le message contient un remerciement, un avis positif, une opinion générale, ou ne signale **aucun problème** → 'label': 'problème non avéré'.\n\n"

    "Si 'label' = 'problème non avéré', mets 'domaine': 'aucun' et 'sous_domaine': 'aucun'.\n\n"

    f"Voici les domaines possibles : {domaines}. "
    f"Voici les sous-domaines possibles : {sous_domaines}. "
    "Analyse attentivement le ton et le contenu du message avant de répondre. "
    "Ta réponse doit uniquement contenir le JSON, sans texte avant ou après."
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
    """
    Classification via LLM avec contexte CLASSIFICATION_CONTEXT.
    Attend un JSON valide (ou pseudo-JSON avec apostrophes).
    """
    context = req.context.strip() or CLASSIFICATION_CONTEXT
    full_text = ""

    # 🔹 Modèle local (Llama.cpp)
    if req.model == "Mistral-7B-Instruct":
        full_prompt = f"{context}\n\nTexte à analyser : {req.prompt}"
        stream = generate_response_stream(model, full_prompt, max_tokens=256)
        full_text = "".join(stream)

    # 🔹 Modèle cloud (Mistral API)
    elif req.model == "Mistral-medium":
        for word in request_model_api(
            prompt=req.prompt,
            context=context,
            model_name="mistral-medium-latest",
        )():
            full_text += word

    else:
        return {"error": "Modèle non reconnu."}

    print(full_text)  # pour debug

    # --- Parsing robuste du JSON ---
    try:
        json_match = re.search(r"\{.*\}", full_text, re.DOTALL)
        if not json_match:
            raise ValueError("JSON non détecté dans la réponse")

        json_str = json_match.group(0)
        json_str = json_str.replace("'", '"')
        json_str = (
            json_str.replace("None", "null")
            .replace("True", "true")
            .replace("False", "false")
        )

        data = json.loads(json_str)

    except Exception as e:
        data = {"error": f"Impossible de parser la réponse : {e}", "raw_response": full_text}

    # --- Ajout du texte original ---
    data["text"] = req.prompt

    # --- Normalisation si problème non avéré ---
    if data.get("label", "").lower() == "problème non avéré":
        data["domaine"] = "aucun"
        data["sous_domaine"] = "aucun"

    # --- Valeurs par défaut si champs manquants ---
    data.setdefault("label", "inconnu")
    data.setdefault("domaine", "inconnu")
    data.setdefault("sous_domaine", "inconnu")

    return data

@app.get("/health")
def health_check():
    return {"status": "ok"}
