"""
Panel de jueces LLM-as-judge (estilo G-Eval).

Dos responsabilidades, alineadas con las dos escalas que conviven:

  1. G-Eval (métrica PRIMARIA, escala 1–5): un PANEL de 2–3 modelos DISTINTOS del
     generador puntúa la CALIDAD del TEXTO DE SALIDA reescrito contra las secciones
     de rúbrica seleccionadas. Se promedia el panel para reducir sesgo de modelo.

  2. Puntaje de RÚBRICA (en puntos, para el umbral y el Gain Score): lo produce UN
     juez consistente (`judge_models[0]`, también distinto del generador) sobre la
     ENTRADA (pre) y la SALIDA (post). pre y post salen SIEMPRE del mismo juez para
     que el Gain de Hake sea válido — nunca del auto-score de un agente.

Todos los jueces se instancian sobre la API de Groq (compatible OpenAI). Reusa el
backoff de `agent_service._ainvoke_with_retry`. Si no hay GROQ_API_KEY o ningún
modelo válido, las funciones lanzan ValueError (los callers lo capturan safe-fail).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from services import rubric_service
from services.agent_service import _ainvoke_with_retry

logger = logging.getLogger(__name__)

LLMInvoke = Callable[[str], Awaitable[str]]

_JUDGE_SYSTEM = (
    "Eres un jurado evaluador metodológico experto e imparcial "
    "(Hernández-Sampieri 2018). Respondes ÚNICAMENTE en JSON válido."
)


# ---------------------------------------------------------------------- #
#  Construcción del panel                                                 #
# ---------------------------------------------------------------------- #

def _judge_models() -> List[str]:
    """Modelos del panel, garantizando que NINGUNO sea el generador (anti-sesgo)."""
    from app.config import settings

    generador = settings.GROQ_MODEL.strip().lower()
    modelos = [m for m in settings.judge_models if m.strip().lower() != generador]
    if not modelos:
        raise ValueError(
            "No hay modelos de juez válidos: GEVAL_JUDGE_MODELS está vacío o todos "
            "coinciden con el generador GROQ_MODEL. Configura modelos distintos."
        )
    return modelos


def _build_judge_llm(model: str):
    """Instancia un juez sobre Groq. temperature=0 para calificación estable."""
    from app.config import settings
    from langchain_openai import ChatOpenAI

    if not settings.GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY no configurada: el panel de jueces G-Eval requiere Groq."
        )
    return ChatOpenAI(
        api_key=settings.GROQ_API_KEY,
        model=model,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.0,
        max_tokens=1500,
        max_retries=0,
    )


def _make_invoker(llm) -> LLMInvoke:
    """Adapta un BaseChatModel a la firma LLMInvoke (prompt str -> respuesta str)."""
    async def _invoke(prompt: str) -> str:
        messages = [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)]
        response = await _ainvoke_with_retry(llm, messages)
        return response.content
    return _invoke


def get_panel_invokers() -> List[Tuple[str, LLMInvoke]]:
    """Lista [(model, invoke)] del panel completo (todos ≠ generador)."""
    return [(m, _make_invoker(_build_judge_llm(m))) for m in _judge_models()]


def get_rubric_invoke() -> Tuple[str, LLMInvoke]:
    """
    Juez ÚNICO y consistente para el puntaje de rúbrica (pre/post + selección de
    secciones). Es `judge_models[0]`. Devuelve (model, invoke).
    """
    model = _judge_models()[0]
    return model, _make_invoker(_build_judge_llm(model))


# ---------------------------------------------------------------------- #
#  Métrica 1 — G-Eval (panel, 1–5) sobre el TEXTO DE SALIDA               #
# ---------------------------------------------------------------------- #

def _build_geval_prompt(salida: str, secciones: List[Dict[str, Any]]) -> str:
    criterios: List[str] = []
    for sec in secciones:
        criterios.append(f"\nSección {sec['numero']} — {sec['nombre']}:")
        for it in sec.get("items", []):
            criterios.append(f"  - {it['criterio']}")
    catalogo = "\n".join(criterios)
    return (
        "Evalúa la CALIDAD del siguiente texto reescrito de un proyecto de tesis, "
        "tomando como marco de calidad los criterios metodológicos indicados.\n\n"
        "Asigna una puntuación GLOBAL de calidad en una escala de 1 a 5:\n"
        "  1 = muy deficiente · 2 = deficiente · 3 = aceptable · "
        "4 = bueno · 5 = excelente.\n\n"
        f"=== CRITERIOS DE REFERENCIA ===\n{catalogo}\n\n"
        f"=== TEXTO REESCRITO A EVALUAR ===\n{salida}\n\n"
        "Responde ÚNICAMENTE con JSON válido de esta forma EXACTA:\n"
        '{\"score\": <entero 1-5>, \"justificacion\": \"<1-2 frases>\"}'
    )


async def geval_quality(
    salida: str,
    secciones: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Métrica 1 (PRIMARIA). Panel de jueces puntúa la calidad de la SALIDA (1–5).
    Devuelve {score (promedio), escala, per_judge:[{model,score,justificacion}], n_judges}.
    Los jueces que fallen se omiten; si fallan todos, score=None.
    """
    panel = get_panel_invokers()

    async def _one(model: str, invoke: LLMInvoke) -> Dict[str, Any]:
        raw = await invoke(_build_geval_prompt(salida, secciones))
        data = rubric_service._parse_json(raw)
        score = data.get("score")
        score = float(score) if score is not None else None
        if score is not None:
            score = max(1.0, min(5.0, score))
        return {"model": model, "score": score, "justificacion": str(data.get("justificacion", "")).strip()}

    resultados = await asyncio.gather(
        *[_one(m, inv) for m, inv in panel], return_exceptions=True
    )
    per_judge: List[Dict[str, Any]] = []
    for r in resultados:
        if isinstance(r, Exception):
            logger.warning(f"geval_quality: un juez falló: {r}")
            continue
        per_judge.append(r)

    validos = [j["score"] for j in per_judge if j["score"] is not None]
    promedio = round(sum(validos) / len(validos), 2) if validos else None
    return {
        "score": promedio,
        "escala": "1-5",
        "per_judge": per_judge,
        "n_judges": len(validos),
    }


# ---------------------------------------------------------------------- #
#  Puntaje de RÚBRICA (mismo juez) — pre/post para umbral y Gain          #
# ---------------------------------------------------------------------- #

async def rubric_pre_post(
    entrada: str,
    salida: str | None,
    secciones: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Puntúa con la rúbrica la ENTRADA (pre) y, si existe, la SALIDA (post), usando
    el MISMO juez (`judge_models[0]`). Devuelve {judge_model, pre, post}. `post`
    es None si no hubo reescritura.
    """
    model, invoke = get_rubric_invoke()

    pre_task = rubric_service.score_text(entrada, secciones, llm_invoke=invoke)
    if salida:
        pre, post = await asyncio.gather(
            pre_task,
            rubric_service.score_text(salida, secciones, llm_invoke=invoke),
        )
    else:
        pre, post = await pre_task, None

    return {"judge_model": model, "pre": pre, "post": post}
