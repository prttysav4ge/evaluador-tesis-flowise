"""
Servicio de agentes secuenciales — modo Python puro (sin Flowise).

Cuando USE_FLOWISE=false en el .env, este servicio ejecuta los 6 agentes
localmente usando LangChain + el LLM configurado.

Ventaja: funciona sin tener Flowise corriendo (ideal para testing inicial).
Desventaja: consume tokens del LLM por cada agente (6 llamadas por query).

Pipeline:
  retrieved_context
       │
       ▼
  [Mentor Intake]  → memory["mentor_intake"]
       │
       ▼
  [Investigador]   → memory["investigador"]
       │
       ▼
  [Auditor]        → memory["auditor"]
       │
       ▼
  [Metodológico]   → memory["metodologico"]
       │
       ▼
  [Redactor]       → memory["redactor"]
       │
       ▼
  [Mentor Final]   → memory["mentor_final"]  ← respuesta final
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

from prompts.agent_prompts import (
    build_auditor_prompt,
    build_investigador_prompt,
    build_mentor_final_prompt,
    build_mentor_intake_prompt,
    build_metodologico_prompt,
    build_redactor_prompt,
    build_texto_sugerido_prompt,
)

logger = logging.getLogger(__name__)


# ====================================================================== #
#  Helpers                                                               #
# ====================================================================== #

def _get_llm() -> BaseChatModel:
    """
    Retorna la instancia del LLM configurado en el .env.
    Soporta OpenAI y Ollama.
    """
    from app.config import settings

    provider = settings.LLM_PROVIDER.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY no configurado. "
                "Agrégalo al archivo .env o cambia LLM_PROVIDER=ollama"
            )
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=0.3,
        )

    elif provider == "ollama":
        try:
            from langchain_ollama import ChatOllama  # paquete nuevo (recomendado)
        except ImportError:
            from langchain_community.chat_models import ChatOllama  # fallback

        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.3,
        )

    else:
        raise ValueError(
            f"LLM_PROVIDER='{provider}' no válido. Usa 'openai' o 'ollama'."
        )


def _parse_json(text: str) -> Dict[str, Any]:
    """
    Extrae y parsea el JSON de la respuesta del LLM.
    Tolerante a texto fuera del JSON (markdown code blocks, etc.).
    """
    # Intenta parsear directo
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Busca bloque ```json ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Busca el JSON más externo con llaves balanceadas
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: devuelve el texto crudo
    logger.warning("⚠️  No se pudo parsear JSON del agente. Retornando texto crudo.")
    return {"raw_output": text, "parse_error": True}


async def _run_agent(
    agent_name: str,
    prompt_text: str,
    llm: BaseChatModel,
) -> Dict[str, Any]:
    """
    Ejecuta un agente: envía el prompt al LLM y parsea la respuesta JSON.
    """
    logger.info(f"🤖 Ejecutando agente: {agent_name}")

    messages = [
        SystemMessage(content="Eres un evaluador académico experto. Responde ÚNICAMENTE en JSON válido."),
        HumanMessage(content=prompt_text),
    ]

    response = await llm.ainvoke(messages)
    result = _parse_json(response.content)

    logger.info(f"✅ Agente '{agent_name}' completado")
    return result


# ====================================================================== #
#  Pipeline principal                                                     #
# ====================================================================== #

async def run_sequential_pipeline(
    question: str,
    retrieved_context: str,
) -> Dict[str, Any]:
    """
    Ejecuta los 6 agentes secuencialmente con memoria acumulativa.

    La memoria se va enriqueciendo con la salida de cada agente.
    Cada agente recibe solo el resumen de los agentes anteriores
    (NO el texto completo de todos los chunks) para ahorrar tokens.

    Returns:
        {
            "question": str,
            "retrieved_context": str,   # primeros 500 chars del contexto
            "memory": {
                "mentor_intake": {...},
                "investigador": {...},
                "auditor": {...},
                "metodologico": {...},
                "redactor": {...},
                "mentor_final": {...}   ← RESPUESTA FINAL
            }
        }
    """
    llm = _get_llm()
    memory: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #  Agente 1 — Mentor Intake                                           #
    # ------------------------------------------------------------------ #
    prompt_1 = build_mentor_intake_prompt(question, retrieved_context)
    memory["mentor_intake"] = await _run_agent("mentor_intake", prompt_1, llm)

    # ------------------------------------------------------------------ #
    #  Agente 2 — Investigador                                            #
    # ------------------------------------------------------------------ #
    prompt_2 = build_investigador_prompt(question, retrieved_context, memory)
    memory["investigador"] = await _run_agent("investigador", prompt_2, llm)

    # ------------------------------------------------------------------ #
    #  Agente 3 — Auditor                                                 #
    # ------------------------------------------------------------------ #
    prompt_3 = build_auditor_prompt(question, retrieved_context, memory)
    memory["auditor"] = await _run_agent("auditor", prompt_3, llm)

    # ------------------------------------------------------------------ #
    #  Agente 4 — Metodológico                                            #
    # ------------------------------------------------------------------ #
    prompt_4 = build_metodologico_prompt(question, retrieved_context, memory)
    memory["metodologico"] = await _run_agent("metodologico", prompt_4, llm)

    # ------------------------------------------------------------------ #
    #  Agente 5 — Redactor                                                #
    # ------------------------------------------------------------------ #
    prompt_5 = build_redactor_prompt(question, retrieved_context, memory)
    memory["redactor"] = await _run_agent("redactor", prompt_5, llm)

    # ------------------------------------------------------------------ #
    #  Agente 6 — Mentor Final (síntesis)                                 #
    # ------------------------------------------------------------------ #
    prompt_6 = build_mentor_final_prompt(question, memory)
    memory["mentor_final"] = await _run_agent("mentor_final", prompt_6, llm)

    return {
        "question": question,
        "retrieved_context_preview": retrieved_context[:500] + "…",
        "memory": memory,
    }


# ====================================================================== #
#  Generador de texto sugerido (post-pipeline, ambos modos)              #
# ====================================================================== #

def _get_texto_llm() -> "BaseChatModel":
    """
    Resuelve el LLM para generar el texto sugerido.

    Orden de prioridad (modo "auto"):
      1. Groq  — si GROQ_API_KEY está configurado (usa el mismo modelo que Flowise)
      2. OpenAI — si OPENAI_API_KEY está configurado
      3. Ollama — siempre disponible como fallback local

    Con LLM_PROVIDER=groq|openai|ollama se fuerza el proveedor sin autodetección.
    """
    from app.config import settings
    from langchain_openai import ChatOpenAI

    provider = settings.LLM_PROVIDER.lower()

    # ── Groq ──────────────────────────────────────────────────────────────
    use_groq = (provider == "groq") or (provider == "auto" and settings.GROQ_API_KEY)
    if use_groq:
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "LLM_PROVIDER=groq pero GROQ_API_KEY no está configurado en .env."
            )
        # Groq expone una API compatible con OpenAI → usamos langchain-openai
        return ChatOpenAI(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.5,
            max_tokens=1500,
        )

    # ── OpenAI ────────────────────────────────────────────────────────────
    use_openai = (provider == "openai") or (provider == "auto" and settings.OPENAI_API_KEY)
    if use_openai:
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER=openai pero OPENAI_API_KEY no está configurado en .env."
            )
        return ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model=settings.OPENAI_MODEL,
            temperature=0.5,
            max_tokens=1500,
        )

    # ── Ollama ────────────────────────────────────────────────────────────
    if provider in ("ollama", "auto"):
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        logger.info(f"Usando Ollama ({settings.OLLAMA_MODEL}) para texto sugerido")
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.5,
        )

    raise ValueError(
        f"LLM_PROVIDER='{provider}' no válido. Usa: auto | groq | openai | ollama"
    )


async def generate_texto_sugerido(
    original_context: str,
    question: str,
    final_evaluation: Dict[str, Any],
    investigador_findings: Dict[str, Any],
) -> str:
    """
    Genera un texto académico mejorado que puede reemplazar la sección
    analizada.  Usa los hallazgos del Investigador para enriquecer el
    contenido y las recomendaciones del Mentor Final para corregirlo.

    Compatible con ambos modos (Flowise y Python puro):
      - En modo Python:  final_evaluation = memory["mentor_final"],
                         investigador_findings = memory["investigador"]
      - En modo Flowise: final_evaluation = JSON final del flujo,
                         investigador_findings = output del nodo Investigador

    Proveedor LLM: Groq (si GROQ_API_KEY configurado) → OpenAI → Ollama.
    """
    llm = _get_texto_llm()

    prompt = build_texto_sugerido_prompt(
        original_context=original_context,
        question=question,
        final_evaluation=final_evaluation,
        investigador_findings=investigador_findings,
    )

    messages = [
        SystemMessage(content=(
            "Eres un experto en redacción académica universitaria en español. "
            "Reescribes secciones de tesis universitarias mejorando su calidad "
            "según evaluaciones de agentes especializados. "
            "Devuelve ÚNICAMENTE el texto mejorado, sin explicaciones ni markdown."
        )),
        HumanMessage(content=prompt),
    ]

    logger.info(
        f"✏️  Generando texto sugerido "
        f"[{type(llm).__name__}]…"
    )
    response = await llm.ainvoke(messages)
    logger.info("✅ Texto sugerido generado")
    return response.content.strip()
