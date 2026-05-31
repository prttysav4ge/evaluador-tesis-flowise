"""
Servicio de agentes secuenciales — modo Python puro (sin Flowise).

Cuando USE_FLOWISE=false en el .env, este servicio ejecuta los 6 agentes
localmente usando LangChain + el LLM configurado.

Ventaja: funciona sin tener Flowise corriendo (ideal para testing inicial).
Desventaja: consume tokens del LLM por cada agente (6 llamadas por query).

Pipeline (los keys de memory se mantienen por compatibilidad con el frontend
y el state de Flowise; las etiquetas visibles ya son las nuevas):
  retrieved_context
       │
       ▼
  [Supervisor]            → memory["mentor_intake"]
       │
       ▼
  [Investigador]          → memory["investigador"]
       │
       ▼
  [Auditor]               → memory["auditor"]
       │
       ▼
  [Metodólogo]            → memory["metodologico"]
       │
       ▼
  [Redactor]              → memory["redactor"]
       │
       ▼
  [Síntesis y Consenso]   → memory["mentor_final"]  ← respuesta final
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List

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
    Soporta Groq, OpenAI y Ollama (y modo 'auto' con detección por claves).

    Orden de prioridad en modo auto:
      1. Groq  — si GROQ_API_KEY está configurado
      2. OpenAI — si OPENAI_API_KEY está configurado
      3. Ollama — fallback local
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
        # Groq expone una API compatible con OpenAI → usamos langchain-openai.
        # max_tokens=800 limita la longitud de cada respuesta JSON de agente.
        # max_retries=0: desactiva el retry interno del SDK; _ainvoke_with_retry
        # lo gestiona con backoff correcto basado en el Retry-After de Groq.
        return ChatOpenAI(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
            max_tokens=800,
            max_retries=0,
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
            temperature=0.3,
            max_tokens=800,
            max_retries=0,
        )

    # ── Ollama ────────────────────────────────────────────────────────────
    if provider in ("ollama", "auto"):
        try:
            from langchain_ollama import ChatOllama  # paquete nuevo (recomendado)
        except ImportError:
            from langchain_community.chat_models import ChatOllama  # fallback

        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.3,
            num_predict=800,   # equivalente a max_tokens para Ollama
        )

    raise ValueError(
        f"LLM_PROVIDER='{provider}' no válido. Usa: auto | groq | openai | ollama"
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


# ── Constantes de retry ────────────────────────────────────────────────────
_LLM_CALL_TIMEOUT  = 30   # segundos máximos por intento individual de LLM
_MAX_RETRIES       = 4    # reintentos ante 429 o timeout (5 intentos en total)
# Si Groq pide esperar más que esto, NO es el límite por-minuto (TPM/RPM) sino el
# límite DIARIO del tier free (o del modelo): esperar no sirve dentro de la
# request. Abortamos rápido con un mensaje claro en vez de dormir minutos y
# reventar el timeout del pipeline.
_MAX_RETRY_WAIT    = 60   # segundos máximos que aceptamos esperar ante un 429


def _parse_wait_seconds(exc: Exception) -> float:
    """
    Extrae el tiempo de espera en segundos del mensaje de error 429 de Groq.

    Groq incluye en el body: "Please try again in 2.47s." o "in 1m30.5s."
    También intenta el header Retry-After si está disponible en la respuesta.
    Retorna 5.0 s como fallback conservador.
    """
    # 1. Intentar parsear desde el mensaje de texto (más fiable en Groq)
    match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", str(exc))
    if match:
        minutes = float(match.group(1) or 0)
        seconds = float(match.group(2))
        return minutes * 60 + seconds + 0.5   # +0.5 s de buffer

    # 2. Header Retry-After (openai.RateLimitError adjunta .response)
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", {})
        for header in ("retry-after", "x-ratelimit-reset-requests"):
            raw = headers.get(header, "")
            if not raw:
                continue
            try:
                return float(raw) + 0.5
            except ValueError:
                pass

    return 5.0   # fallback


async def _ainvoke_with_retry(
    llm: BaseChatModel,
    messages: List,
) -> Any:
    """
    Llama a llm.ainvoke con reintentos para:
      - openai.RateLimitError / 429  (Groq TPM/RPM)
      - asyncio.TimeoutError         (LLM lento o colgado)

    Cada intento tiene su propio timeout de _LLM_CALL_TIMEOUT segundos.
    El tiempo de espera entre reintentos se lee del mensaje de Groq
    ("Please try again in Xs") para respetar el ventana de rate-limit exacta.
    """
    import openai as _openai   # import local para no depender en el nivel de módulo
    from app.config import settings

    last_exc: Exception | None = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=_LLM_CALL_TIMEOUT,
            )

        except asyncio.TimeoutError as exc:
            last_exc = exc
            if attempt == _MAX_RETRIES:
                logger.error(
                    f"⏰ LLM no respondió en {_LLM_CALL_TIMEOUT}s tras "
                    f"{_MAX_RETRIES + 1} intentos. Abortando."
                )
                raise
            wait = 3.0 * (2 ** attempt)
            logger.warning(
                f"⏰ Timeout LLM (intento {attempt + 1}/{_MAX_RETRIES + 1}). "
                f"Reintentando en {wait:.1f}s…"
            )
            await asyncio.sleep(wait)

        except _openai.RateLimitError as exc:
            last_exc = exc
            wait = _parse_wait_seconds(exc)

            # Espera excesiva ⇒ límite DIARIO del tier free / del modelo. Esperar
            # dentro de la request no tiene sentido (bloquea minutos y revienta el
            # timeout). Fallar rápido con un mensaje accionable.
            if wait > _MAX_RETRY_WAIT:
                logger.error(
                    f"❌ Groq pide esperar {wait:.0f}s — es el límite DIARIO del tier "
                    f"free (o del modelo {settings.GROQ_MODEL}), no el por-minuto. "
                    "Abortando rápido. Soluciones: cambia GROQ_MODEL a uno con cuota "
                    "(p.ej. llama-3.1-8b-instant), usa otra API key, o sube al Dev Tier."
                )
                raise

            if attempt == _MAX_RETRIES:
                logger.error(
                    f"❌ Groq 429 persistente tras {_MAX_RETRIES + 1} intentos. "
                    "Considera actualizar al Dev Tier en console.groq.com/settings/billing "
                    "o reducir top_k en la consulta."
                )
                raise

            logger.warning(
                f"⚠️  Groq 429 Rate Limit "
                f"(intento {attempt + 1}/{_MAX_RETRIES + 1}). "
                f"Esperando {wait:.1f}s (Retry-After de Groq)…"
            )
            await asyncio.sleep(wait)

    raise last_exc  # type: ignore[misc]  — nunca debería alcanzarse


async def _run_agent(
    agent_name: str,
    prompt_text: str,
    llm: BaseChatModel,
) -> Dict[str, Any]:
    """
    Ejecuta un agente: envía el prompt al LLM y parsea la respuesta JSON.
    Los reintentos por 429 y los timeouts los gestiona _ainvoke_with_retry.
    """
    logger.info(f"🤖 Ejecutando agente: {agent_name}")

    messages = [
        SystemMessage(content="Eres un evaluador académico experto. Responde ÚNICAMENTE en JSON válido."),
        HumanMessage(content=prompt_text),
    ]

    response = await _ainvoke_with_retry(llm, messages)
    result = _parse_json(response.content)

    logger.info(f"✅ Agente '{agent_name}' completado")
    return result


# ====================================================================== #
#  Pipeline principal                                                     #
# ====================================================================== #

async def run_sequential_pipeline(
    question: str,
    retrieved_context: str,
    reference_context: str = "",
    previous_iteration: str | None = None,
) -> Dict[str, Any]:
    """
    Ejecuta los 6 agentes secuencialmente con memoria acumulativa.

    Args:
        retrieved_context: fragmentos relevantes del PDF de tesis (RAG primario).
        reference_context: fragmentos de la Biblioteca Metodológica (RAG cruzado).
        previous_iteration: síntesis de la iteración anterior (JSON string).
            Vacía en la primera iteración del panel. Cuando esté presente, el
            agente Síntesis la usa para refinar en lugar de empezar de cero.

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
    #  Agente 2 — Investigador (con Biblioteca cruzada)                   #
    # ------------------------------------------------------------------ #
    prompt_2 = build_investigador_prompt(
        question, retrieved_context, memory, reference_context=reference_context
    )
    memory["investigador"] = await _run_agent("investigador", prompt_2, llm)

    # ------------------------------------------------------------------ #
    #  Agente 3 — Auditor                                                 #
    # ------------------------------------------------------------------ #
    prompt_3 = build_auditor_prompt(question, retrieved_context, memory)
    memory["auditor"] = await _run_agent("auditor", prompt_3, llm)

    # ------------------------------------------------------------------ #
    #  Agente 4 — Metodológico (con Biblioteca cruzada — agente clave)    #
    # ------------------------------------------------------------------ #
    prompt_4 = build_metodologico_prompt(
        question, retrieved_context, memory, reference_context=reference_context
    )
    memory["metodologico"] = await _run_agent("metodologico", prompt_4, llm)

    # ------------------------------------------------------------------ #
    #  Agente 5 — Redactor                                                #
    # ------------------------------------------------------------------ #
    prompt_5 = build_redactor_prompt(question, retrieved_context, memory)
    memory["redactor"] = await _run_agent("redactor", prompt_5, llm)

    # ------------------------------------------------------------------ #
    #  Agente 6 — Síntesis y Consenso (con iteración previa si aplica)    #
    # ------------------------------------------------------------------ #
    prompt_6 = build_mentor_final_prompt(
        question, memory, previous_iteration=previous_iteration
    )
    memory["mentor_final"] = await _run_agent("mentor_final", prompt_6, llm)

    return {
        "question": question,
        "retrieved_context_preview": retrieved_context[:500] + "…",
        "memory": memory,
    }


# ====================================================================== #
#  Generador de texto sugerido (post-pipeline, ambos modos)              #
# ====================================================================== #

# Modelo de respaldo para la ÚLTIMA llamada (texto sugerido). El modelo
# principal (p.ej. llama-3.3-70b) suele quedarse sin cuota TPM del minuto
# porque el pipeline de 6 agentes ya la consumió; este modelo vive en un
# bucket de rate-limit SEPARADO, así que responde aunque el 70b esté en 429.
_TEXTO_FALLBACK_MODEL     = "llama-3.1-8b-instant"
# El bucket del 8b-instant es chico (~6000 TPM free tier). Recortamos el
# contexto original para que la llamada de respaldo quepa sin volver a 429.
_TEXTO_FALLBACK_CTX_CHARS = 6000


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
        # max_retries=0: los reintentos los gestiona _ainvoke_with_retry
        return ChatOpenAI(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.5,
            max_tokens=1500,
            max_retries=0,
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
            max_retries=0,
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
    from app.config import settings

    system_msg = SystemMessage(content=(
        "Eres un experto en redacción académica universitaria en español. "
        "Reescribes secciones de tesis universitarias mejorando su calidad "
        "según evaluaciones de agentes especializados. "
        "Devuelve ÚNICAMENTE el texto mejorado, sin explicaciones ni markdown."
    ))

    def _build_messages(ctx: str) -> List:
        prompt = build_texto_sugerido_prompt(
            original_context=ctx,
            question=question,
            final_evaluation=final_evaluation,
            investigador_findings=investigador_findings,
        )
        return [system_msg, HumanMessage(content=prompt)]

    # ── Intento principal (modelo configurado, p.ej. 70b) ────────────────
    llm = _get_texto_llm()
    try:
        logger.info(f"✏️  Generando texto sugerido [{settings.GROQ_MODEL}]…")
        response = await _ainvoke_with_retry(llm, _build_messages(original_context))
        logger.info("✅ Texto sugerido generado")
        return response.content.strip()
    except Exception as primary_exc:
        # El modelo principal casi siempre falla por 429 (su cuota TPM del minuto
        # ya la quemó el pipeline). Reintentamos con un modelo de bucket separado
        # y contexto recortado. Solo aplica con Groq y si el principal no ES ya el
        # de respaldo (en cuyo caso reintentar el mismo modelo no aportaría nada).
        already_fallback = settings.GROQ_MODEL.lower() == _TEXTO_FALLBACK_MODEL.lower()
        if not settings.GROQ_API_KEY or already_fallback:
            raise

        from langchain_openai import ChatOpenAI
        logger.warning(
            f"⚠️  Texto sugerido falló con {settings.GROQ_MODEL} "
            f"({type(primary_exc).__name__}: {primary_exc}). "
            f"Reintentando con fallback {_TEXTO_FALLBACK_MODEL} (bucket separado)…"
        )
        fallback_llm = ChatOpenAI(
            api_key=settings.GROQ_API_KEY,
            model=_TEXTO_FALLBACK_MODEL,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.5,
            max_tokens=1200,
            max_retries=0,
        )
        ctx = original_context[:_TEXTO_FALLBACK_CTX_CHARS]
        response = await _ainvoke_with_retry(fallback_llm, _build_messages(ctx))
        logger.info(f"✅ Texto sugerido generado (fallback {_TEXTO_FALLBACK_MODEL})")
        return response.content.strip()
