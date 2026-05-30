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

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from vectorstore.chroma_store import chroma_store
from vectorstore.refs_store import refs_store

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
    iterations: int = Field(
        default=1,
        ge=1,
        le=3,
        description=(
            "Cantidad de iteraciones del panel multiagente. En cada iteración, "
            "el agente de Síntesis recibe la síntesis previa y la refina."
        ),
    )


class QueryResponse(BaseModel):
    question: str
    mode: str  # "flowise" | "python_agents"
    chunks_retrieved: int
    elapsed_seconds: float
    context_preview: str
    reference_chunks_retrieved: int = 0
    reference_context_preview: str  = ""
    iterations_count: int           = 1
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

    logger.info(f"📚 Fragmentos recuperados (tesis): {len(raw_results)}")

    # ------------------------------------------------------------------ #
    #  2b. Retrieval cruzado contra Biblioteca Metodológica               #
    # ------------------------------------------------------------------ #
    # Recuperamos fragmentos de los libros metodológicos usando la misma
    # pregunta. Si la biblioteca está vacía o falla, el pipeline sigue
    # funcionando con solo el contexto de la tesis (degradación graciosa).
    refs_raw: list[Dict[str, Any]] = []
    reference_context: str = ""
    try:
        if refs_store.collection.count() > 0:
            refs_raw = refs_store.query(body.question, top_k=_REFS_TOP_K)
            reference_context = _format_refs_context(refs_raw)
            logger.info(f"📖 Fragmentos recuperados (biblioteca): {len(refs_raw)}")
    except Exception as exc:
        logger.warning(f"⚠️  Retrieval de biblioteca falló (continuando sin refs): {exc}")
        refs_raw = []
        reference_context = ""

    reference_context_preview = (
        reference_context[:300] + "…" if len(reference_context) > 300 else reference_context
    )

    # ------------------------------------------------------------------ #
    #  3. Agentes — loop de iteraciones                                   #
    # ------------------------------------------------------------------ #
    # En cada iteración corremos el panel completo (6 agentes). A partir
    # de la iteración 2, la síntesis previa se pasa como contexto extra
    # al agente Síntesis para que refine en vez de empezar de cero.
    iterations_history: list[Dict[str, Any]] = []
    previous_iteration_text: Optional[str]   = None
    final_result: Dict[str, Any]             = {}
    mode: str                                = "python_agents"

    for iter_num in range(1, body.iterations + 1):
        logger.info(f"🔁 Iteración {iter_num}/{body.iterations}")

        if settings.USE_FLOWISE:
            iter_result, iter_mode = await _call_flowise_with_fallback(
                body.question, retrieved_context, reference_context,
                body.session_id, previous_iteration=previous_iteration_text,
            )
        else:
            iter_mode = "python_agents"
            iter_result = await _call_python_agents(
                body.question, retrieved_context, reference_context,
                previous_iteration=previous_iteration_text,
            )

        # Extraemos el output de la síntesis de esta iteración para alimentar
        # la siguiente. JSON string compacto para minimizar tokens.
        iter_synthesis = _extract_synthesis_json(iter_result)
        previous_iteration_text = iter_synthesis if iter_synthesis else None

        iterations_history.append({
            "iteration": iter_num,
            "mode": iter_mode,
            "result": iter_result,
        })
        final_result = iter_result
        mode = iter_mode

    # El frontend espera el último iter como top-level (backward compat con
    # las 4 pestañas) y la historia completa en 'iterations_history' para
    # poder renderizar P2 con sesiones múltiples.
    result = final_result
    result["iterations_history"] = [
        {
            "iteration": h["iteration"],
            "mode": h["mode"],
            "memory": h["result"].get("memory"),
            "flowise_response": h["result"].get("flowise_response"),
            "_flowise_fallback": h["result"].get("_flowise_fallback"),
        }
        for h in iterations_history
    ]

    # Adjuntamos contexto cruzado al resultado para que el frontend lo
    # use en la sub-pestaña 'De libros de referencia' / 'Contexto cruzado'.
    result["reference_context"] = reference_context
    result["reference_chunks"]  = [
        {
            "text":   r.get("text", ""),
            "source": r.get("metadata", {}).get("source", "?"),
            "page":   r.get("metadata", {}).get("page", "?"),
            "score":  r.get("score"),
        }
        for r in refs_raw
    ]

    # ------------------------------------------------------------------ #
    #  4. Generar texto sugerido (post-pipeline, ambos modos)            #
    #     Se limita a 45 s para no superar el timeout total del cliente. #
    # ------------------------------------------------------------------ #
    _TEXTO_SUGERIDO_TIMEOUT = 45  # segundos máximos para esta llamada extra
    try:
        from services.agent_service import generate_texto_sugerido
        evaluation_data       = _extract_evaluation_data(result)
        investigador_findings = _extract_investigador_findings(result)
        texto_sugerido = await asyncio.wait_for(
            generate_texto_sugerido(
                original_context=retrieved_context,
                question=body.question,
                final_evaluation=evaluation_data,
                investigador_findings=investigador_findings,
            ),
            timeout=_TEXTO_SUGERIDO_TIMEOUT,
        )
        result["texto_sugerido"]    = texto_sugerido
        result["original_context"]  = retrieved_context   # para comparación en UI
    except asyncio.TimeoutError:
        logger.warning(
            f"⚠️  generate_texto_sugerido excedió {_TEXTO_SUGERIDO_TIMEOUT}s — "
            "se omite en esta respuesta."
        )
        result["texto_sugerido"]   = None
        result["original_context"] = retrieved_context
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
        reference_chunks_retrieved=len(refs_raw),
        reference_context_preview=reference_context_preview,
        iterations_count=body.iterations,
        result=result,
    )


# ====================================================================== #
#  Helpers privados                                                       #
# ====================================================================== #

# Cantidad de fragmentos de la Biblioteca Metodológica a recuperar por consulta.
# Más bajo que el TOP_K de la tesis para no inflar el prompt de los agentes.
_REFS_TOP_K = 3


def _format_refs_context(refs_results: list) -> str:
    """
    Formatea los chunks recuperados de la Biblioteca Metodológica con
    atribución (libro + página) en lugar de 'sección detectada'.
    """
    if not refs_results:
        return ""
    parts: list = []
    for i, r in enumerate(refs_results, 1):
        meta = r.get("metadata", {}) or {}
        source = meta.get("source", "?")
        page   = meta.get("page", "?")
        parts.append(
            f"[Biblioteca | Fragmento {i} | Libro: {source} | p.{page}]\n"
            f"{r.get('text', '')}"
        )
    return "\n\n---\n\n".join(parts)


def _extract_synthesis_json(result: Dict[str, Any]) -> str:
    """
    Extrae el JSON de la síntesis final (Mentor Final / Síntesis y Consenso)
    como string compacto para pasarlo a la siguiente iteración.

    Funciona en ambos modos:
      - Flowise: result['flowise_response']['text'] suele ser el JSON.
      - Python:  result['memory']['mentor_final'] es el dict.
    """
    # Modo Python
    if "memory" in result:
        synth = result["memory"].get("mentor_final")
        if synth:
            try:
                return json.dumps(synth, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                return ""

    # Modo Flowise
    if "flowise_response" in result:
        flow_resp = result["flowise_response"]
        if isinstance(flow_resp, dict):
            text = flow_resp.get("text") or flow_resp.get("output") or ""
            if isinstance(text, str) and text.strip():
                return text.strip()

    return ""


_FLOWISE_FILE_ERROR = "filePath"   # señal de nodo con archivo local roto en cloud

# Errores transitorios de infraestructura (Flowise Cloud / Cloudflare) que
# justifican un fallback automático a los agentes Python:
#   - 502 Bad Gateway        → upstream caído
#   - 503 Service Unavailable → upstream saturado
#   - 504 Gateway Timeout    → upstream lento (caso común con agentflow + 6 LLMs)
_FLOWISE_TRANSIENT_HTTP = ("HTTP 502", "HTTP 503", "HTTP 504")


def _is_transient_flowise_error(exc_msg: str) -> bool:
    """True si el error de Flowise es un fallo de infraestructura recuperable."""
    if any(code in exc_msg for code in _FLOWISE_TRANSIENT_HTTP):
        return True
    # Cloudflare / nginx devuelven HTML como cuerpo de error
    if "<!DOCTYPE html" in exc_msg or "<html" in exc_msg:
        return True
    return False


async def _call_flowise_with_fallback(
    question: str,
    context: str,
    reference_context: str,
    session_id: Optional[str],
    previous_iteration: Optional[str] = None,
) -> tuple[Dict[str, Any], str]:
    """
    Intenta llamar a Flowise. Si devuelve un error recuperable (nodo con
    archivo local roto, timeout 504, gateway caído), hace fallback automático
    a los agentes Python. Retorna (result_dict, mode_str).

    Args:
        reference_context: contexto de la Biblioteca Metodológica (libros).
            Se inyecta en el JSON payload que parsea el CustomFunction de
            Flowise para que los agentes lo lean desde el Flow State.
        previous_iteration: JSON string de la síntesis de la iteración anterior.
            Vacío en la primera iteración. La Síntesis lo usa para refinar
            en lugar de empezar de cero.
    """
    from flowise.client import flowise_client

    try:
        response = await flowise_client.call_chatflow(
            question=question,
            context=context,
            reference_context=reference_context,
            session_id=session_id,
            previous_iteration=previous_iteration,
        )
        return {"flowise_response": response}, "flowise"

    except ValueError as exc:
        exc_msg = str(exc)

        # Flowise 500 por nodo con ruta de archivo local que no existe en cloud
        if _FLOWISE_FILE_ERROR in exc_msg:
            logger.warning(
                "⚠️  Flowise devolvió 500 (filePath undefined). "
                "El chatflow tiene un nodo Document Loader con archivo local "
                "que no existe en Flowise Cloud. "
                "Haciendo fallback a agentes Python automáticamente."
            )
            result = await _call_python_agents(question, context, reference_context, previous_iteration=previous_iteration)
            result["_flowise_fallback"] = (
                "Flowise Cloud falló (nodo con archivo local roto). "
                "Se usaron los agentes Python como fallback."
            )
            return result, "python_agents_fallback"

        # Errores transitorios de infraestructura (504/502/503, HTML de Cloudflare)
        if _is_transient_flowise_error(exc_msg):
            logger.warning(
                "⚠️  Flowise Cloud devolvió error transitorio (timeout/gateway). "
                "Haciendo fallback a agentes Python automáticamente."
            )
            result = await _call_python_agents(question, context, reference_context, previous_iteration=previous_iteration)
            result["_flowise_fallback"] = (
                "Flowise Cloud no respondió a tiempo (504/502/503). "
                "Se usaron los agentes Python como fallback."
            )
            return result, "python_agents_fallback"

        # Cualquier otro error de Flowise → propagar como 502
        logger.exception(f"Error llamando a Flowise [ValueError]: {exc_msg}")
        raise HTTPException(
            status_code=502,
            detail=(
                f"[ValueError] {exc_msg} — "
                "Verifica que Flowise esté corriendo y que FLOWISE_CHATFLOW_ID sea correcto. "
                "Puedes cambiar USE_FLOWISE=false en .env para usar los agentes Python directamente."
            ),
        )

    except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError) as exc:
        # Timeout o conexión caída del cliente HTTP → fallback automático
        exc_type = type(exc).__name__
        logger.warning(
            f"⚠️  Flowise Cloud inaccesible ({exc_type}). "
            "Haciendo fallback a agentes Python automáticamente."
        )
        result = await _call_python_agents(question, context, reference_context)
        result["_flowise_fallback"] = (
            f"Flowise Cloud inaccesible ({exc_type}). "
            "Se usaron los agentes Python como fallback."
        )
        return result, "python_agents_fallback"

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
    reference_context: str = "",
    previous_iteration: Optional[str] = None,
) -> Dict[str, Any]:
    from services.agent_service import run_sequential_pipeline

    try:
        return await run_sequential_pipeline(
            question, context, reference_context, previous_iteration=previous_iteration
        )
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
