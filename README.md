# SAVIA-API-LLM

API locale pour exposer un modèle LLM (Mistral 7B, TinyLLaMA...) via FastAPI + Prometheus (metrics).

## 🚀 Lancer en local (GPU)

```bash
docker build -t savia-api-llm .
docker run --gpus all -p 8000:8000 --env-file .env savia-api-llm
