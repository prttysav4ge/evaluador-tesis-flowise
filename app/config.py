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
    # Groq → OpenAI → Ollama (en ese orden de prioridad).
    # Puedes forzar uno con LLM_PROVIDER=groq|openai|ollama.
    LLM_PROVIDER: str = "auto"   # "auto" | "groq" | "openai" | "ollama"

    # Groq (recomendado: usa los mismos modelos que Flowise, sin coste extra)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"   # mismo modelo que el Agentflow

    # OpenAI
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

    # ----- CHUNKING -----
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
