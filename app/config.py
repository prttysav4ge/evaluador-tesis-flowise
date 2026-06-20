"""
Configuración central del proyecto via variables de entorno.
Carga automáticamente el archivo .env
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # ----- SERVIDOR -----
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # ----- LLM -----
    # El proveedor se elige automáticamente según las claves disponibles:
    # OpenAI → Ollama (en ese orden de prioridad).
    # Puedes forzar uno con LLM_PROVIDER=openai|ollama.
    LLM_PROVIDER: str = "auto"   # "auto" | "openai" | "ollama"

    # OpenAI (proveedor principal)
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"

    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # ----- FLOWISE -----
    FLOWISE_URL: str = "http://localhost:3000"
    FLOWISE_CHATFLOW_ID: str = ""
    FLOWISE_API_KEY: Optional[str] = None
    # True  = envía la pregunta+contexto a Flowise y deja que sus agentes respondan
    # False = corre los agentes secuenciales directamente en Python (recomendado para tests)
    USE_FLOWISE: bool = False
    # Límite de caracteres del contexto RAG enviado a Flowise.
    # Los nodos LLM del Agentflow tienen maxTokens bajos (300-350); con contexto
    # demasiado grande el JSON de salida queda truncado y el flujo falla.
    # ~1 500 chars ≈ 375 tokens ≈ presupuesto seguro con 1-2 fragmentos completos.
    FLOWISE_MAX_CONTEXT_CHARS: int = 1500

    # ----- EMBEDDINGS -----
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"

    # ----- CHROMADB -----
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "academic_thesis"

    # ----- RAG -----
    TOP_K: int = 5

    # ----- JUEZ G-EVAL / RÚBRICA -----
    # Panel de jueces LLM-as-judge. DEBEN ser modelos DISTINTOS del generador
    # (OPENAI_MODEL) para evitar sesgo. Lista separada por comas; se instancian
    # todos sobre la API de OpenAI. Confirmar IDs vigentes en
    # platform.openai.com/docs/models si alguno deja de estar disponible.
    GEVAL_JUDGE_MODELS: str = (
        "gpt-4o,gpt-4.1-mini,gpt-4.1-nano"
    )
    # Umbral (proporción 0-1) del puntaje de rúbrica de la ENTRADA por encima del
    # cual NO se reescribe (solo se recomienda pulir). Orientativo: 0.90.
    REDACTOR_THRESHOLD: float = 0.90
    # Métrica condicional (5). Solo válida con >=2 iteraciones equivalentes.
    # Se deja escrita pero desactivada hasta cerrar las decisiones abiertas.
    ENABLE_ITERATIVE_CONSISTENCY: bool = False

    @property
    def judge_models(self) -> list[str]:
        """GEVAL_JUDGE_MODELS parseado a lista, sin vacíos ni duplicados (orden estable)."""
        seen: dict[str, None] = {}
        for m in self.GEVAL_JUDGE_MODELS.split(","):
            m = m.strip()
            if m and m not in seen:
                seen[m] = None
        return list(seen.keys())

    # ----- CHUNKING -----
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
