"""
Métricas NLP — Sprint 3.

Compara el texto original analizado (la sección de tesis recuperada del PDF)
contra el texto sugerido por el pipeline (output del Redactor / generate_texto_sugerido).

Las métricas se calculan en el frontend Streamlit (no en el pipeline LLM) para
no añadir latencia al request principal. Se invocan post-render desde la
Pestaña 4 (Reportes).

Métricas implementadas:
  - ROUGE-1, ROUGE-2, ROUGE-L (f-measure) — overlap de unigramas, bigramas, LCS.
  - BLEU                                  — n-gram precision con brevity penalty.
  - Similitud coseno                      — sobre embeddings multilingual-e5.
  - Gain Score (Hake)                     — mejora normalizada de puntaje.
  - Cohen's Kappa                         — placeholder (requiere 2 evaluadores).

Estrategia de fallback: cada métrica es safe-fail. Si una librería no está
instalada o el cómputo arroja excepción, devolvemos None en ese campo
(la UI lo renderiza como "—") en vez de propagar el error.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
#  Métricas individuales                                                  #
# ---------------------------------------------------------------------- #

def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """ROUGE-1/2/L f-measure (rango 0-1). Mayor = mayor overlap léxico."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def compute_bleu(reference: str, hypothesis: str) -> float:
    """BLEU corpus normalizado a [0,1]. Mayor = más fluidez vs referencia."""
    from sacrebleu import corpus_bleu

    bleu = corpus_bleu([hypothesis], [[reference]])
    return round(bleu.score / 100.0, 4)


def compute_cosine_similarity(text1: str, text2: str) -> float:
    """
    Similitud coseno [0,1] sobre embeddings multilingual-e5 (ya normalizados).
    Reutiliza el embedder singleton del backend para no pagar la carga de
    modelo dos veces.
    """
    import numpy as np

    from embeddings.embedder import embedder

    vectors = embedder.embed_documents([text1, text2])
    v1, v2  = np.array(vectors[0]), np.array(vectors[1])
    # embed_documents normaliza los vectores → cos sim = dot product directo
    return round(float(np.dot(v1, v2)), 4)


def compute_gain_score(
    score_before: float,
    score_after: float,
    max_score: float = 10.0,
) -> float:
    """
    Ganancia normalizada de Hake: g = (post - pre) / (max - pre).
    Rango [-1, 1]. Positivo = mejora; negativo = retroceso.
    """
    denom = max_score - score_before
    if denom <= 0:
        return 0.0
    return round((score_after - score_before) / denom, 4)


def compute_kappa(ratings_a: list, ratings_b: list) -> Optional[float]:
    """
    Cohen's Kappa entre 2 evaluadores. Placeholder por ahora — requiere que
    el pipeline genere dos evaluaciones independientes (ej. dos pasadas del
    panel con seeds distintos). Se habilitará cuando el loop nativo de Flowise
    (Sprint 4) permita comparar iteración N vs iteración N+1.
    """
    return None


# ---------------------------------------------------------------------- #
#  Agregador                                                              #
# ---------------------------------------------------------------------- #

def compute_all(
    reference: str,
    hypothesis: str,
    score_before: Optional[float] = None,
    score_after:  Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calcula todas las métricas comparando reference (original) vs hypothesis
    (sugerido). Safe-fail: cada métrica fallida queda en None en lugar de
    abortar el cómputo entero.

    Returns:
        {
            "rouge1": float | None,
            "rouge2": float | None,
            "rougeL": float | None,
            "bleu":   float | None,
            "cosine_similarity": float | None,
            "gain_score":       float | None,   # solo si se pasan ambos scores
            "kappa":            None,            # placeholder
        }
    """
    metrics: Dict[str, Any] = {}

    try:
        metrics.update(compute_rouge(reference, hypothesis))
    except Exception as exc:
        logger.warning(f"compute_rouge falló: {exc}")
        metrics["rouge1"] = metrics["rouge2"] = metrics["rougeL"] = None

    try:
        metrics["bleu"] = compute_bleu(reference, hypothesis)
    except Exception as exc:
        logger.warning(f"compute_bleu falló: {exc}")
        metrics["bleu"] = None

    try:
        metrics["cosine_similarity"] = compute_cosine_similarity(reference, hypothesis)
    except Exception as exc:
        logger.warning(f"compute_cosine_similarity falló: {exc}")
        metrics["cosine_similarity"] = None

    if score_before is not None and score_after is not None:
        try:
            metrics["gain_score"] = compute_gain_score(score_before, score_after)
        except Exception as exc:
            logger.warning(f"compute_gain_score falló: {exc}")
            metrics["gain_score"] = None
    else:
        metrics["gain_score"] = None

    metrics["kappa"] = compute_kappa([], [])
    return metrics
