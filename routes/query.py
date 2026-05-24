"""
Ruta: POST /api/v1/query

Orquesta el pipeline RAG + agentes:
  1. Recupera chunks relevantes desde ChromaDB (RAG)
  2. Formatea el contexto
  3a. Si USE_FLOWISE=true  → llama al chatflow de Flowise
  3b. Si USE_FLOWISE=false → ejecuta los 6 agentes en Python directamente
  4. Retorna la respuesta final al frontend
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from vectorstore.chroma_store import chroma_store

logger = logging.getLogger(__name__)
router = APIRouter()


# ====================================================================== #
#  Modelos de entrada / salida                                            #
# ====================================================================== #

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Pregunta o instrucción de evaluación sobre la tesis.",
        example="evalúa la formulación del problema de investigación",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Número de fragmentos relevantes a recuperar de ChromaDB.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="ID de sesión para mantener conversación en Flowise (opcional).",
    )


class QueryResponse(BaseModel):
    question: str
    mode: str  # "flowise" | "python_agents"
    chunks_retrieved: int
    elapsed_seconds: float
    context_preview: str
    result: Dict[str, Any]


# ====================================================================== #
#  Endpoint principal                                                     #
# ====================================================================== #

@router.post(
    "/query",
    summary="Consultar la tesis y ejecutar evaluación multiagente",
    response_model=QueryResponse,
)
async def query_thesis(body: QueryRequest) -> QueryResponse:
    """
    Pipeline RAG + agentes secuenciales.

    **Flujo:**
    - Recupera los chunks más relevantes de ChromaDB para la pregunta.
    - Si `USE_FLOWISE=true`: envía contexto + pregunta a Flowise.
    - Si `USE_FLOWISE=false`: ejecuta los 6 agentes secuenciales en Python.

    **Ejemplo de pregunta:**
    - `"evalúa la formulación del problema de investigación"`
    - `"¿es adecuado el marco metodológico?"`
    - `"¿qué debilidades tiene el marco teórico?"`
    """
    start = time.time()

    # ------------------------------------------------------------------ #
    #  1. Verificar que hay datos en ChromaDB                             #
    # ------------------------------------------------------------------ #
    info = chroma_store.get_info()
    if info["total_chunks"] == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                "No hay ninguna tesis cargada. "
                "Sube primero un PDF en POST /api/v1/upload-pdf."
            ),
        )

    # ------------------------------------------------------------------ #
    #  2. Retrieval desde ChromaDB                                        #
    # ------------------------------------------------------------------ #
    logger.info(f"🔍 Buscando chunks relevantes para: '{body.question[:80]}…'")

    try:
        raw_results = chroma_store.query(body.question, top_k=body.top_k)
    except Exception as exc:
        logger.exception("Error en retrieval ChromaDB")
        raise HTTPException(status_code=500, detail=f"Error en ChromaDB: {str(exc)}")

    if not raw_results:
        raise HTTPException(
            status_code=404,
            detail="No se encontraron fragmentos relevantes. Revisa la pregunta o sube más contenido.",
        )

    retrieved_context = chroma_store.format_context(raw_results)
    context_preview = retrieved_context[:300] + "…" if len(retrieved_context) > 300 else retrieved_context

    logger.info(f"📚 Fragmentos recuperados: {len(raw_results)}")

    # ------------------------------------------------------------------ #
    #  3. Agentes                                                          #
    # ------------------------------------------------------------------ #
    if settings.USE_FLOWISE:
        mode = "flowise"
        result = await _call_flowise(body.question, retrieved_context, body.session_id)
    else:
        mode = "python_agents"
        result = await _call_python_agents(body.question, retrieved_context)

    # ------------------------------------------------------------------ #
    #  4. Generar texto sugerido (post-pipeline, ambos modos)            #
    # ------------------------------------------------------------------ #
    try:
        from services.agent_service import generate_texto_sugerido
        evaluation_data       = _extract_evaluation_data(result)
        investigador_findings = _extract_investigador_findings(result)
        texto_sugerido = await generate_texto_sugerido(
            original_context=retrieved_context,
            question=body.question,
            final_evaluation=evaluation_data,
            investigador_findings=investigador_findings,
        )
        result["texto_sugerido"]    = texto_sugerido
        result["original_context"]  = retrieved_context   # para comparación en UI
    except Exception as exc:
        logger.warning(f"⚠️  No se pudo generar texto sugerido: {exc}")
        result["texto_sugerido"]   = None
        result["original_context"] = retrieved_context

    elapsed = round(time.time() - start, 2)
    logger.info(f"✅ Query completada en {elapsed}s (modo: {mode})")

    return QueryResponse(
        question=body.question,
        mode=mode,
        chunks_retrieved=len(raw_results),
        elapsed_seconds=elapsed,
        context_preview=context_preview,
        result=result,
    )


# ====================================================================== #
#  Helpers privados                                                       #
# ====================================================================== #

async def _call_flowise(
    question: str,
    context: str,
    session_id: Optional[str],
) -> Dict[str, Any]:
    from flowise.client import flowise_client

    try:
        response = await flowise_client.call_chatflow(
            question=question,
            context=context,
            session_id=session_id,
        )
        return {"flowise_response": response}
    except Exception as exc:
        exc_type = type(exc).__name__
        exc_msg  = str(exc) or "(sin mensaje — probablemente timeout)"
        logger.exception(f"Error llamando a Flowise [{exc_type}]: {exc_msg}")
        raise HTTPException(
            status_code=502,
            detail=(
                f"[{exc_type}] {exc_msg} — "
                "Verifica que Flowise esté corriendo y que FLOWISE_CHATFLOW_ID sea correcto. "
                "Puedes cambiar USE_FLOWISE=false en .env para usar los agentes Python directamente."
            ),
        )


async def _call_python_agents(
    question: str,
    context: str,
) -> Dict[str, Any]:
    from services.agent_service import run_sequential_pipeline

    try:
        return await run_sequential_pipeline(question, context)
    except Exception as exc:
        logger.exception("Error en pipeline de agentes Python")
        raise HTTPException(
            status_code=500,
            detail=f"Error en agentes: {str(exc)}",
        )


def _extract_evaluation_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae el dict de evaluación final del resultado del pipeline.

    - Modo Flowise:  parsea result["flowise_response"]["text"] como JSON.
    - Modo Python:   retorna result["memory"]["mentor_final"].
    """
    # Modo Flowise
    if "flowise_response" in result:
        flowise_resp = result["flowise_response"]
        if isinstance(flowise_resp, dict):
            text = flowise_resp.get("text", "")
            if text:
                try:
                    return json.loads(text.strip())
                except Exception:
                    pass
        return {}

    # Modo Python
    if "memory" in result:
        return result["memory"].get("mentor_final", {})

    return {}


def _extract_investigador_findings(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrae los hallazgos del agente Investigador del resultado.

    - Modo Python:   result["memory"]["investigador"]  (directo).
    - Modo Flowise:  busca el nodo 'Investigador' en agentFlowExecutedData
                     y parsea su output.content como JSON.
    """
    # Modo Python
    if "memory" in result:
        return result["memory"].get("investigador", {})

    # Modo Flowise: buscar en el árbol de ejecución
    if "flowise_response" in result:
        flow_data = result["flowise_response"]
        if isinstance(flow_data, dict):
            exec_data = flow_data.get("agentFlowExecutedData", [])
            for node in exec_data:
                if not isinstance(node, dict):
                    continue
                label = node.get("nodeLabel", "").lower()
                if "investigador" in label or "investigat" in label:
                    content = (
                        node.get("data", {})
                            .get("output", {})
                            .get("content", "")
                    )
                    if content:
                        try:
                            return json.loads(content)
                        except Exception:
                            pass

    return {}
