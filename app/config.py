# app/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Chargement du .env
load_dotenv()

# =======================
# Paramètres bruts
# =======================
MODEL_NAME = os.getenv("MODEL_NAME")                     # ex: models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
MODEL_FILE = os.getenv("MODEL_FILE")                    # ex: mistral-7b-instruct-v0.2.Q4_K_M.gguf
DEVICE = os.getenv("DEVICE", "auto").lower()            # "cpu", "gpu", "auto"
USE_VLLM = os.getenv("USE_VLLM", "false").lower() == "true"
PORT = int(os.getenv("PORT", "8000"))
API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_MISTRAL = os.getenv("MODEL_MISTRAL")
# =======================
# Résolution du chemin
# =======================
base_model_dir = Path(__file__).parent.parent / "models"

# Priorité : MODEL_NAME > MODEL_FILE > défaut
if MODEL_NAME:
    model_path = Path(MODEL_NAME)
elif MODEL_FILE:
    model_path = base_model_dir / MODEL_FILE
else:
    raise ValueError("❌ Aucun modèle spécifié (ni MODEL_NAME ni MODEL_FILE). Vérifie ton .env")

# Absolutise le chemin
MODEL_PATH = str(model_path.resolve())

