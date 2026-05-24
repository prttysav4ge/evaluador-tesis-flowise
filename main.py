"""
POC: Evaluador de Tesis Universitarias — Sistema RAG Multiagente
================================================================

Arquitectura:
  Frontend → FastAPI → PDF Ingestion → ChromaDB
                    → RAG Retrieval → Flowise / Python Agents → Respuesta

Endpoints principales:
  POST /api/v1/upload-pdf   — Subir y procesar un PDF
  POST /api/v1/query        — Consultar y evaluar con agentes
  GET  /api/v1/health       — Estado del sistema
  GET  /api/v1/collection   — Info de ChromaDB
  DELETE /api/v1/collection — Reiniciar colección

Docs interactivos: http://localhost:8000/docs
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

# ====================================================================== #
#  Logging                                                                #
# ====================================================================== #
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ====================================================================== #
#  Lifespan — inicialización al arrancar el servidor                      #
# ====================================================================== #
@asynccontextmanager
async def lifespan(_app: FastAPI):  # noqa: ARG001 — FastAPI requiere este parámetro en la firma
    """
    Se ejecuta una vez al inicio (antes de recibir requests).
    Inicializa ChromaDB y pre-carga el modelo de embeddings en memoria.
    """
    logger.info("=" * 60)
    logger.info("🚀 Iniciando POC — Evaluador de Tesis RAG Multiagente")
    logger.info("=" * 60)
    logger.info(f"   LLM Provider  : {settings.LLM_PROVIDER.upper()}")
    logger.info(f"   Embedding     : {settings.EMBEDDING_MODEL}")
    logger.info(f"   ChromaDB dir  : {settings.CHROMA_PERSIST_DIR}")
    logger.info(f"   Flowise URL   : {settings.FLOWISE_URL}")
    logger.info(f"   Modo agentes  : {'FLOWISE' if settings.USE_FLOWISE else 'PYTHON DIRECTO'}")
    logger.info("-" * 60)

    # Inicializar ChromaDB
    from vectorstore.chroma_store import chroma_store
    chroma_store.initialize()

    # Pre-cargar el modelo de embeddings (evita el cold-start en el primer request)
    logger.info("⏳ Pre-cargando modelo de embeddings…")
    from embeddings.embedder import embedder
    embedder._load_model()
    logger.info("✅ Sistema listo para recibir requests")
    _display_host = "localhost" if settings.HOST in ("0.0.0.0", "::") else settings.HOST
    logger.info(f"📖 Docs: http://{_display_host}:{settings.PORT}/docs")
    logger.info("=" * 60)

    yield  # ← el servidor está corriendo entre aquí y el return

    logger.info("🛑 Servidor detenido correctamente")


# ====================================================================== #
#  Aplicación FastAPI                                                     #
# ====================================================================== #
app = FastAPI(
    title="🎓 Evaluador de Tesis — RAG Multiagente",
    description=(
        "POC para evaluación académica de tesis universitarias.\n\n"
        "**Flujo de uso:**\n"
        "1. Sube tu tesis con `POST /api/v1/upload-pdf`\n"
        "2. Consulta con `POST /api/v1/query`\n"
        "3. El sistema recupera fragmentos relevantes y los envía a los agentes\n"
        "4. Los 6 agentes evalúan secuencialmente con memoria acumulativa\n\n"
        "**Stack:** FastAPI · ChromaDB · multilingual-e5-small · "
        "LangChain · Flowise · OpenAI/Ollama"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — permite llamadas desde cualquier frontend local
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================================== #
#  Routers                                                                #
# ====================================================================== #
from routes.upload import router as upload_router
from routes.query import router as query_router
from routes.admin import router as admin_router

app.include_router(upload_router, prefix="/api/v1", tags=["📥 Upload PDF"])
app.include_router(query_router, prefix="/api/v1", tags=["🔍 Query & Agentes"])
app.include_router(admin_router, prefix="/api/v1", tags=["⚙️ Admin"])


# ====================================================================== #
#  Root                                                                   #
# ====================================================================== #
@app.get("/", tags=["Root"], include_in_schema=False)
async def root():
    return {
        "app": "Evaluador de Tesis RAG Multiagente",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "upload_pdf": "POST /api/v1/upload-pdf",
            "query": "POST /api/v1/query",
            "health": "GET /api/v1/health",
            "collection_info": "GET /api/v1/collection",
            "reset_collection": "DELETE /api/v1/collection?confirm=true",
            "list_chunks": "GET /api/v1/chunks",
        },
    }


# ====================================================================== #
#  Entry point                                                            #
# ====================================================================== #
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
