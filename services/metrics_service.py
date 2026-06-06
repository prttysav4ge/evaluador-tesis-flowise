"""
Métricas de evaluación — EXACTAMENTE 4 (más 1 condicional desactivada).

El stack anterior (ROUGE, BLEU, Cohen's Kappa, y el Gain alimentado por el
auto-score de un agente) fue eliminado por inválido. Las métricas vigentes son:

  1. LLM-as-judge (G-Eval, escala 1–5) — PRIMARIA, calidad del TEXTO DE SALIDA.
     Vive en `judge_service.geval_quality` (es un LLM, no un cálculo determinista).
  2. Gain Score (Hake) — PROCESO. g = (post − pre)/(máx − pre). pre/post salen del
     MISMO juez de rúbrica (`judge_service.rubric_pre_post`), nunca de un agente.
  3. Cosine Similarity — GUARDRAIL semántico entrada vs salida (no mide calidad).
  4. Context Precision (sin referencia) — componente RAG sobre los chunks de los
     LIBROS. Vive en `context_precision.context_precision_without_reference`.

Condicional (5) Iterative Consistency: solo válida con ≥2 iteraciones equivalentes;
se deja escrita pero desactivada por `settings.ENABLE_ITERATIVE_CONSISTENCY`.

Las métricas deterministas (cosine, gain, consistencia) viven aquí; las que
necesitan LLM (G-Eval, context precision) viven en sus módulos. La orquestación
y el ensamblado del payload final ocurren en `routes/query.py`.

Estrategia safe-fail: cada función captura sus errores y devuelve None.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
#  3. Cosine Similarity — guardrail semántico                             #
# ---------------------------------------------------------------------- #

def compute_cosine_similarity(text1: str, text2: str) -> Optional[float]:
    """
    Similitud coseno [0,1] sobre embeddings multilingual-e5 (ya normalizados).
    GUARDRAIL, no calidad: muy alto ⇒ casi no reescribió; muy bajo ⇒ se desvió
    del sentido original. Reutiliza el embedder singleton del backend.
    """
    try:
        import numpy as np
        from embeddings.embedder import embedder

        vectors = embedder.embed_documents([text1, text2])
        v1, v2 = np.array(vectors[0]), np.array(vectors[1])
        # embed_documents normaliza → cos sim = dot product directo.
        return round(float(np.dot(v1, v2)), 4)
    except Exception as exc:
        logger.warning(f"compute_cosine_similarity falló: {exc}")
        return None


# Umbrales orientativos para interpretar el guardrail en la UI.
COSINE_ALARM_HIGH = 0.97   # casi idéntico → probablemente no reescribió
COSINE_ALARM_LOW = 0.60    # demasiado distinto → posible desvío de sentido


def cosine_guardrail_flag(cos: Optional[float]) -> str:
    """Etiqueta de alarma para el guardrail coseno: 'ok' | 'alto' | 'bajo' | 'n/a'."""
    if cos is None:
        return "n/a"
    if cos >= COSINE_ALARM_HIGH:
        return "alto"
    if cos <= COSINE_ALARM_LOW:
        return "bajo"
    return "ok"


# ---------------------------------------------------------------------- #
#  2. Gain Score (Hake)                                                   #
# ---------------------------------------------------------------------- #

def compute_gain_score(
    pre: float,
    post: float,
    max_score: float = 1.0,
) -> Optional[float]:
    """
    Ganancia normalizada de Hake: g = (post − pre) / (máx − pre).

    `pre` y `post` deben ser puntajes de rúbrica del MISMO juez (proporciones
    0–1 con max_score=1.0, o puntos con max_score=máximo de la sección). Rango
    [-1, 1]: positivo = mejora; 0 si la entrada ya estaba en el máximo.
    """
    try:
        denom = max_score - pre
        if denom <= 0:
            return 0.0
        return round((post - pre) / denom, 4)
    except Exception as exc:
        logger.warning(f"compute_gain_score falló: {exc}")
        return None


# ---------------------------------------------------------------------- #
#  4. Context Precision (delegado al módulo dedicado)                     #
# ---------------------------------------------------------------------- #

async def compute_context_precision(
    question: str,
    answer: str,
    ref_chunks: List[Dict[str, Any]],
    *,
    llm_invoke,
) -> Optional[float]:
    """Thin wrapper sobre context_precision_without_reference (variante sin referencia)."""
    from services.context_precision import context_precision_without_reference

    try:
        return await context_precision_without_reference(
            question, answer, ref_chunks, llm_invoke=llm_invoke
        )
    except Exception as exc:
        logger.warning(f"compute_context_precision falló: {exc}")
        return None


# ---------------------------------------------------------------------- #
#  5. Iterative Consistency — CONDICIONAL, desactivada por defecto        #
# ---------------------------------------------------------------------- #

def compute_iteration_consistency(scores: List[float]) -> Optional[float]:
    """
    Consistencia entre iteraciones: proporción de puntajes dentro de ±1.0 del
    promedio. Rango [0,1].

    CONDICIONAL: solo es válida si el flujo corrió ≥2 iteraciones equivalentes.
    Controlada por `settings.ENABLE_ITERATIVE_CONSISTENCY` (False por defecto);
    los callers deben respetar ese flag antes de exponerla.
    """
    if not scores or len(scores) < 2:
        return None
    avg = sum(scores) / len(scores)
    within = sum(1 for s in scores if abs(s - avg) <= 1.0) / len(scores)
    return round(within, 4)
