"""
Rutas de administración y diagnóstico.

  GET  /api/v1/health        — estado del sistema
  GET  /api/v1/collection    — info de ChromaDB
  DELETE /api/v1/collection  — reinicia la colección (borra todo)
  GET  /api/v1/chunks        — lista los primeros N chunks almacenados
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from vectorstore.chroma_store import chroma_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", summary="Estado del sistema")
async def health() -> Dict[str, Any]:
    """
    Devuelve el estado de todos los componentes:
    ChromaDB, modelo de embeddings, Flowise (si está configurado).
    """
    from flowise.client import flowise_client

    # Check Flowise (no bloqueante)
    flowise_ok = False
    if settings.FLOWISE_CHATFLOW_ID:
        flowise_ok = await flowise_client.health_check()

    db_info = chroma_store.get_info()

    return {
        "status": "healthy",
        "components": {
            "chromadb": {
                "status": "connected",
                "chunks_stored": db_info["total_chunks"],
                "collection": db_info["collection"],
            },
            "embeddings": {
                "model": settings.EMBEDDING_MODEL,
                "status": "ready",
            },
            "llm": {
                "provider": settings.LLM_PROVIDER,
                "model": (
                    settings.OPENAI_MODEL
                    if settings.LLM_PROVIDER == "openai"
                    else settings.OLLAMA_MODEL
                ),
            },
            "flowise": {
                "url": settings.FLOWISE_URL,
                "chatflow_id": settings.FLOWISE_CHATFLOW_ID or "not_configured",
                "reachable": flowise_ok,
                "mode": "active" if settings.USE_FLOWISE else "disabled",
            },
        },
        "execution_mode": "flowise" if settings.USE_FLOWISE else "python_agents",
    }


@router.get("/collection", summary="Información de la colección ChromaDB")
async def get_collection_info() -> Dict[str, Any]:
    """Devuelve estadísticas de la colección ChromaDB activa."""
    return chroma_store.get_info()


@router.delete(
    "/collection",
    summary="⚠️ Reiniciar colección (BORRA TODOS LOS CHUNKS)",
)
async def reset_collection(
    confirm: bool = Query(
        False,
        description="Debes pasar confirm=true para ejecutar el borrado.",
    ),
) -> Dict[str, Any]:
    """
    **DESTRUCTIVO**: Elimina todos los chunks almacenados en ChromaDB.

    Requiere el parámetro `?confirm=true` como protección.
    Útil cuando quieres cargar una nueva tesis desde cero.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Operación cancelada. Pasa '?confirm=true' para confirmar el borrado.",
        )

    chroma_store.reset()
    return {
        "success": True,
        "message": "✅ Colección reiniciada. Ahora puedes subir un nuevo PDF.",
    }


@router.get("/chunks", summary="Listar chunks almacenados (preview)")
async def list_chunks(
    limit: int = Query(10, ge=1, le=100, description="Número máximo de chunks a retornar"),
    offset: int = Query(0, ge=0, description="Desplazamiento"),
) -> Dict[str, Any]:
    """
    Retorna una muestra de los chunks almacenados en ChromaDB.
    Útil para verificar que el PDF se procesó correctamente.
    """
    try:
        result = chroma_store.collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    docs: List[str] = result.get("documents") or []
    metas: List[Dict] = result.get("metadatas") or []

    chunks = [
        {
            "preview": doc[:200] + "…" if len(doc) > 200 else doc,
            "metadata": meta,
        }
        for doc, meta in zip(docs, metas)
    ]

    return {
        "total_in_collection": chroma_store.get_info()["total_chunks"],
        "returned": len(chunks),
        "offset": offset,
        "chunks": chunks,
    }
