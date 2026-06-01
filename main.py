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
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings

# ====================================================================== #
#  Logging                                                                #
# ====================================================================== #
# El root SIEMPRE en INFO: si lo pusiéramos en DEBUG, librerías de terceros
# (pdfminer token-por-token, watchdog evento-por-evento, chromadb) inundan el
# arranque con miles de líneas y revientan el startup de Streamlit Cloud
# ("Error running app"). El flag DEBUG solo sube el nivel de NUESTROS módulos.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
_app_level = logging.DEBUG if settings.DEBUG else logging.INFO
for _app in ("main", "app", "services", "routes", "vectorstore", "embeddings",
             "flowise"):
    logging.getLogger(_app).setLevel(_app_level)
# Terceros especialmente verbosos: aún más callados, ni siquiera INFO.
for _noisy in ("pdfminer", "pdfplumber", "watchdog", "chromadb",
               "httpcore", "httpx", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Guardamos referencias fuertes a las tareas de fondo: asyncio solo mantiene
# referencias débiles y podría recolectarlas a media ejecución.
_BG_TASKS: set[asyncio.Task] = set()


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

    # Inicializar ChromaDB (colección de tesis)
    from vectorstore.chroma_store import chroma_store
    chroma_store.initialize()

    # Inicializar colección de libros metodológicos (Biblioteca Metodológica).
    from vectorstore.refs_store import refs_store, index_reference_books
    refs_store.initialize()

    # Pre-cargar el modelo de embeddings (evita el cold-start en el primer request)
    logger.info("⏳ Pre-cargando modelo de embeddings…")
    from embeddings.embedder import embedder
    embedder._load_model()

    # Auto-index de la Biblioteca Metodológica si la colección está vacía.
    # En producción (Streamlit Cloud) esta es la única forma de poblar la
    # colección — los PDFs viajan commiteados en reference_books/.
    #
    # NO BLOQUEANTE: pdfplumber procesa cientos de páginas (TOC-aware) y genera
    # miles de embeddings; tarda minutos. Si bloqueáramos el lifespan, Streamlit
    # Cloud mata el arranque por timeout ("Error running app"). Por eso lo
    # lanzamos en un hilo de fondo: la app empieza a servir de inmediato y la
    # Biblioteca se va poblando. Las consultas degradan con gracia mientras
    # tanto (routes/query.py ya chequea collection.count() > 0 antes de usarla).
    refs_count = refs_store.collection.count()
    if refs_count == 0:
        async def _bg_index_biblioteca() -> None:
            try:
                logger.info("📚 Biblioteca vacía — auto-indexando en SEGUNDO PLANO…")
                added = await asyncio.to_thread(index_reference_books)
                logger.info(f"📚 Auto-index de fondo completado: {added} chunks agregados")
            except Exception as exc:
                logger.exception(f"⚠️  Auto-index de biblioteca falló: {exc}")

        _task = asyncio.create_task(_bg_index_biblioteca())
        _BG_TASKS.add(_task)
        _task.add_done_callback(_BG_TASKS.discard)
        logger.info("📚 Biblioteca en indexado de fondo; la app ya acepta requests.")
    else:
        logger.info(f"📚 Biblioteca ya indexada: {refs_count} chunks")
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
from routes.reference_books import router as refs_router

app.include_router(upload_router, prefix="/api/v1", tags=["📥 Upload PDF"])
app.include_router(query_router, prefix="/api/v1", tags=["🔍 Query & Agentes"])
app.include_router(admin_router, prefix="/api/v1", tags=["⚙️ Admin"])
app.include_router(refs_router, prefix="/api/v1", tags=["📚 Biblioteca"])


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
