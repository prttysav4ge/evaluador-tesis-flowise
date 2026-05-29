"""
Cliente HTTP para la API de Flowise.

Flowise v2 diferencia chatflows de agentflows en endpoints distintos:
  - Chatflows: POST /api/v1/prediction/{chatflowId}          (sin auth si es público)
  - Agentflows: POST /api/v1/agentflows/{agentflowId}/prediction  (requiere API key)

El Agentflow existente tiene un CustomFunction (initializeFlowState)
que parsea el campo `question` como JSON con las siguientes keys:
  - section_type:       tipo de sección académica (o "rag_query")
  - section_text:       la pregunta/instrucción del evaluador
  - retrieved_context:  fragmentos relevantes recuperados de ChromaDB
  - research_line:      línea de investigación (opcional)
  - match_type:         tipo de match RAG (opcional)

El Python backend serializa un dict como JSON string y lo envía en `question`.
El CustomFunction lo parsea y lo distribuye al Flow State para los agentes LLM.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class FlowiseClient:
    """
    Wrapper sobre la API HTTP de Flowise.
    Formatea el payload para que sea compatible con el CustomFunction
    initializeFlowState del Agentflow existente.
    """

    def __init__(self) -> None:
        from app.config import settings

        self.base_url = settings.FLOWISE_URL.rstrip("/")
        self.chatflow_id = settings.FLOWISE_CHATFLOW_ID
        self.api_key = settings.FLOWISE_API_KEY
        # 90 s — si Flowise Cloud no responde aquí, mejor caer al fallback Python.
        # Esperar 5 min bloquea innecesariamente al cliente cuando el cloud tiene
        # problemas (504 / Cloudflare gateway timeouts son comunes).
        self.timeout = 90.0

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _truncate_context(self, context: str) -> str:
        """
        Recorta el contexto RAG a FLOWISE_MAX_CONTEXT_CHARS para evitar que
        los nodos LLM del Agentflow (maxTokens 300-350) produzcan JSON truncado.

        Estrategia: corta en el límite del último fragmento completo que quepa
        (usa el separador '---' que genera format_context). Si no hay separador
        o el primer fragmento ya excede el límite, trunca en el carácter N.
        """
        from app.config import settings

        max_chars = settings.FLOWISE_MAX_CONTEXT_CHARS
        if len(context) <= max_chars:
            return context

        # Intentar preservar fragmentos completos
        separator = "\n\n---\n\n"
        fragments = context.split(separator)
        kept: List[str] = []   # List[str] funciona en Python 3.8+
        total = 0
        for frag in fragments:
            if total + len(frag) + len(separator) > max_chars:
                break
            kept.append(frag)
            total += len(frag) + len(separator)

        if kept:
            truncated = separator.join(kept)
            logger.warning(
                f"⚠️  Contexto RAG truncado: {len(context)} → {len(truncated)} chars "
                f"({len(fragments) - len(kept)} fragmento(s) descartado(s)). "
                "Ajusta FLOWISE_MAX_CONTEXT_CHARS en .env para incluir más."
            )
            return truncated

        # Si ni siquiera el primer fragmento cabe, truncar en el carácter N
        logger.warning(
            f"⚠️  Contexto RAG truncado en {max_chars} chars (fragmento único muy grande)."
        )
        return context[:max_chars]

    def _build_question_payload(
        self,
        question: str,
        context: str,
        reference_context: str = "",
    ) -> str:
        """
        Construye el JSON string que el CustomFunction `initializeFlowState`
        espera parsear desde `$flow.input`.

        El CustomFunction hace:
            data = JSON.parse($flow.input)
            → data.section_type        → $flow.state.current_section_type
            → data.section_text        → $flow.state.student_input
            → data.retrieved_context   → $flow.state.retrieved_context    ← RAG tesis
            → data.reference_context   → $flow.state.reference_context    ← RAG biblioteca
            → data.research_line       → $flow.state.validated_research_line
            → data.match_type          → $flow.state.research_line_match_type
        """
        # Truncamos el reference_context con un cap más bajo que el de la tesis
        # (60% del cap principal) para no inflar demasiado el prompt total.
        # Los agentes que lo usen igual deben tratarlo como contexto secundario.
        from app.config import settings
        refs_cap = max(int(settings.FLOWISE_MAX_CONTEXT_CHARS * 0.6), 600)

        payload_data = {
            "section_type":      "rag_query",
            "section_text":      question,
            "retrieved_context": self._truncate_context(context),
            "reference_context": (
                reference_context[:refs_cap] if reference_context else ""
            ),
            "research_line":     "",
            "match_type":        "semantic_similarity",
        }
        return json.dumps(payload_data, ensure_ascii=False)

    async def call_chatflow(
        self,
        question: str,
        context: str,
        reference_context: str = "",
        session_id: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Llama al Agentflow de Flowise con la pregunta y el contexto RAG.

        El campo `question` se serializa como JSON para que el CustomFunction
        `initializeFlowState` pueda distribuirlo al Flow State del Agentflow.

        El flujo en Flowise queda así:
          Start → CustomFunction (parsea JSON) → Mentor Intake → Investigador
                → Auditor → Metodológico → Redactor → Mentor Final → End

        Returns:
            dict con la respuesta de Flowise (incluye key 'text' con la respuesta final).
        """
        if not self.chatflow_id:
            raise ValueError(
                "FLOWISE_CHATFLOW_ID no está configurado. "
                "Copia el ID desde la URL del Agentflow en Flowise."
            )

        # El Agentflow espera el question como JSON serializado como string
        question_json = self._build_question_payload(question, context, reference_context)

        payload: Dict[str, Any] = {
            "question": question_json,
            "streaming": False,   # fuerza respuesta JSON simple; evita SSE
        }

        if session_id:
            payload["sessionId"] = session_id

        if override_config:
            payload["overrideConfig"] = override_config

        # Flowise v3: el endpoint unificado para chatflows Y agentflows es /api/v1/prediction/{id}
        url = f"{self.base_url}/api/v1/prediction/{self.chatflow_id}"
        logger.info(f"📡 Llamando a Flowise Agentflow: {url}")
        logger.debug(f"   section_type: rag_query | context_chars: {len(context)}")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers=self._headers(),
            )

        # ── Loguear siempre para facilitar depuración ────────────────────
        ct = response.headers.get("content-type", "desconocido")
        logger.info(
            f"Flowise HTTP {response.status_code} | content-type: {ct} "
            f"| body: {len(response.content)} bytes"
        )

        # ── Verificar código de estado ────────────────────────────────────
        if response.status_code >= 400:
            body_preview = response.text[:300]
            raise ValueError(
                f"Flowise devolvió HTTP {response.status_code}. "
                f"Respuesta: {body_preview!r}"
            )

        # ── Parsear respuesta ─────────────────────────────────────────────
        raw_text = response.text

        # Intento 1: JSON directo
        try:
            data = json.loads(raw_text)
            logger.info("✅ Respuesta JSON de Flowise recibida correctamente")
            return data
        except json.JSONDecodeError:
            pass

        # Intento 2: Server-Sent Events (SSE) — Flowise puede devolver este formato
        # si streaming estaba habilitado en el flujo a pesar de streaming:False
        sse_data = self._parse_sse_response(raw_text)
        if sse_data is not None:
            logger.info("✅ Respuesta SSE de Flowise parseada como JSON")
            return sse_data

        # Sin solución: lanzar error descriptivo
        raise ValueError(
            f"La respuesta de Flowise no es JSON válido ni SSE reconocible. "
            f"Content-Type: {ct}. "
            f"Primeros 300 chars: {raw_text[:300]!r}"
        )

    @staticmethod
    def _parse_sse_response(text: str) -> Optional[Dict[str, Any]]:
        """
        Parsea una respuesta SSE (Server-Sent Events) de Flowise.
        Busca la última línea 'data: {...}' y la retorna como dict.
        Retorna None si el texto no tiene formato SSE.
        """
        # Extraer todas las líneas "data: <payload>"
        data_lines = [
            m.group(1)
            for m in re.finditer(r"^data:\s*(.+)$", text, re.MULTILINE)
            if m.group(1).strip() not in ("[DONE]", "")
        ]
        if not data_lines:
            return None

        # Recorrer en orden inverso buscando un JSON con "text" o "output"
        for line in reversed(data_lines):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and ("text" in obj or "output" in obj):
                    return obj
            except json.JSONDecodeError:
                continue

        # Último recurso: el último data válido como JSON
        for line in reversed(data_lines):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue

        return None

    async def health_check(self) -> bool:
        """Verifica que el servidor Flowise esté levantado y la API key sea válida."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(
                    f"{self.base_url}/api/v1/agentflows/{self.chatflow_id}",
                    headers=self._headers(),
                )
                return r.status_code == 200
        except Exception:
            return False


# Instancia singleton
flowise_client = FlowiseClient()
