"""
Context Precision (variante SIN referencia) — componente RAG.

Reimplementa `llm_context_precision_without_reference` de Ragas sin añadir la
librería (que engordaría el deploy en Streamlit Cloud). Mide si los fragmentos
recuperados de los LIBROS metodológicos indexados son relevantes para responder
la consulta — opera sobre los chunks, no sobre el texto de la tesis.

Fórmula (average precision @K con verdicts binarios del LLM):
    Precision@k = (#relevantes en top-k) / k
    AP = Σ_k (Precision@k · v_k) / Σ_k v_k        (0 si no hay relevantes)
donde v_k ∈ {0,1} es el verdict de relevancia del fragmento en la posición k
(orden de recuperación, mejor primero). Rango [0,1]; mayor = los fragmentos
relevantes están mejor rankeados.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

LLMInvoke = Callable[[str], Awaitable[str]]


def _build_verdict_prompt(question: str, answer: str, chunks: List[str]) -> str:
    numbered = "\n\n".join(
        f"[Fragmento {i}]\n{c}" for i, c in enumerate(chunks, 1)
    )
    return (
        "Dada una CONSULTA y una RESPUESTA, decide para cada FRAGMENTO recuperado "
        "si fue ÚTIL/RELEVANTE para llegar a la respuesta (1) o no (0).\n\n"
        f"=== CONSULTA ===\n{question}\n\n"
        f"=== RESPUESTA ===\n{answer}\n\n"
        f"=== FRAGMENTOS RECUPERADOS ===\n{numbered}\n\n"
        "Responde ÚNICAMENTE con JSON válido de esta forma EXACTA:\n"
        '{\"verdicts\": [{\"fragmento\": 1, \"relevante\": 0|1}, ...]}\n'
        "Incluye un verdict por cada fragmento, en orden."
    )


def average_precision(verdicts: List[int]) -> float:
    """
    Average precision @K a partir de verdicts binarios ordenados por ranking.
    Determinista → unidad testeable sin LLM.
    """
    if not verdicts or sum(verdicts) == 0:
        return 0.0
    relevantes_acum = 0
    suma = 0.0
    for k, v in enumerate(verdicts, 1):
        if v:
            relevantes_acum += 1
            suma += relevantes_acum / k   # Precision@k en las posiciones relevantes
    return round(suma / sum(verdicts), 4)


async def context_precision_without_reference(
    question: str,
    answer: str,
    ref_chunks: List[Dict[str, Any]],
    *,
    llm_invoke: LLMInvoke,
) -> Optional[float]:
    """
    Calcula Context Precision sobre los fragmentos de la Biblioteca (libros).

    Args:
        ref_chunks: lista de dicts con al menos la clave 'text' (orden de
            recuperación = mejor primero), p.ej. result['reference_chunks'].

    Returns:
        float [0,1], o None si no hay fragmentos o el juez falla.
    """
    from services import rubric_service

    textos = [str(c.get("text", "")).strip() for c in ref_chunks if c.get("text")]
    if not textos:
        return None

    try:
        raw = await llm_invoke(_build_verdict_prompt(question, answer, textos))
        data = rubric_service._parse_json(raw)
    except Exception as exc:
        logger.warning(f"context_precision: el juez falló: {exc}")
        return None

    # Mapea verdicts por número de fragmento (1-indexado); ausente = 0.
    by_idx: Dict[int, int] = {}
    for v in data.get("verdicts", []):
        try:
            idx = int(v.get("fragmento"))
            by_idx[idx] = 1 if int(v.get("relevante", 0)) else 0
        except (TypeError, ValueError):
            continue

    verdicts = [by_idx.get(i, 0) for i in range(1, len(textos) + 1)]
    return average_precision(verdicts)
