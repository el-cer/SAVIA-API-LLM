from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import FastAPI
from prometheus_client import make_asgi_app

REQUEST_COUNT = Counter("llm_requests_total", "Total LLM requests")
LATENCY = Histogram("llm_request_latency_seconds", "Latency of LLM requests")

def setup_metrics(app: FastAPI):
    app.mount("/metrics", make_asgi_app())
