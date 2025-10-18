from llama_cpp import Llama
from app.config import MODEL_PATH

def load_model():
    print(f"[LOADING] 🔁 GGUF model loading from: {MODEL_PATH}")

    model = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        use_mlock=True,
        n_gpu_layers=-1,  # ← jusqu'à 32 pour Mistral-7B avec ta 1070 Ti
        verbose=True,
    )

    print("[LOADING] ✅ GGUF model loaded.")
    return model
