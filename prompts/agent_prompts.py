"""
Prompts de los 6 agentes especializados + generador de texto sugerido.

Criterios de diseño:
  - Salida SIEMPRE en JSON válido (sin texto extra), salvo build_texto_sugerido_prompt
    que pide texto académico plano listo para pegar en la tesis.
  - Bajo consumo de tokens: reciben sólo lo necesario de la memoria acumulada
  - Cada agente tiene un ROL único y claro
  - El Mentor Final sintetiza todo para el estudiante
"""
from __future__ import annotations
import json
from typing import Any, Dict


# ====================================================================== #
#  AGENTE 1 — Mentor Intake (evaluación inicial)                         #
# ====================================================================== #

def build_mentor_intake_prompt(question: str, context: str) -> str:
    return f"""Eres el MENTOR DE EVALUACIÓN INICIAL de tesis universitarias.

ROL: Realizar la evaluación inicial de la pregunta del evaluador sobre la tesis.

=== CONTEXTO RECUPERADO DE LA TESIS ===
{context}

=== PREGUNTA DEL EVALUADOR ===
{question}

=== INSTRUCCIONES ===
1. Lee el contexto recuperado de la tesis con atención.
2. Identifica el tema central y la sección académica presente.
3. Evalúa si el contexto recuperado es suficiente para responder la pregunta.
4. Identifica los 3 aspectos clave que deben evaluarse.
5. Señala limitaciones del contexto recuperado (si las hay).

RESPONDE ÚNICAMENTE en formato JSON válido, sin texto adicional antes ni después:
{{
  "tema_identificado": "tema central de la tesis en 1 oración",
  "seccion_relevante": "nombre de la sección académica identificada",
  "pertinencia_pregunta": "alta|media|baja",
  "contexto_suficiente": true,
  "aspectos_clave": ["aspecto1", "aspecto2", "aspecto3"],
  "evaluacion_inicial": "evaluación concisa del fragmento en 2-3 oraciones",
  "limitaciones_contexto": ["limitacion1"],
  "flags": []
}}"""


# ====================================================================== #
#  AGENTE 2 — Investigador (análisis de calidad investigativa)           #
# ====================================================================== #

def build_investigador_prompt(
    question: str,
    context: str,
    memory: Dict[str, Any],
    reference_context: str = "",
) -> str:
    mentor_summary = json.dumps(memory.get("mentor_intake", {}), ensure_ascii=False)
    refs_block = (
        f"\n=== BIBLIOTECA METODOLÓGICA (libros de referencia) ===\n{reference_context}\n"
        if reference_context else ""
    )
    refs_instr = (
        "6. Si la Biblioteca aporta principios relevantes, citalos al respaldar tus "
        "observaciones (ej. 'según Hernández Sampieri...'). Privilegia la coincidencia "
        "entre lo que dice la tesis y lo que recomienda la literatura metodológica.\n"
        if reference_context else ""
    )
    return f"""Eres el AGENTE INVESTIGADOR especializado en análisis de investigación académica.

ROL: Analizar la calidad investigativa del fragmento de tesis.

=== PREGUNTA ===
{question}

=== CONTEXTO DE LA TESIS ===
{context}
{refs_block}
=== EVALUACIÓN PREVIA (Mentor Intake) ===
{mentor_summary}

=== INSTRUCCIONES ===
1. Analiza la solidez de la argumentación e investigación.
2. Evalúa si hay respaldo teórico y bibliográfico.
3. Identifica fortalezas y debilidades investigativas concretas.
4. Sugiere 2-3 mejoras específicas y realizables.
5. Asigna una puntuación de 0 a 10.
{refs_instr}
RESPONDE ÚNICAMENTE en formato JSON válido:
{{
  "fortalezas": ["fortaleza1", "fortaleza2"],
  "debilidades": ["debilidad1", "debilidad2"],
  "respaldo_teorico": "adecuado|parcial|insuficiente",
  "relevancia_cientifica": "alta|media|baja",
  "sugerencias": ["sugerencia1", "sugerencia2"],
  "puntuacion": 7.5,
  "comentario": "análisis investigativo en 2-3 oraciones",
  "biblioteca_aplicada": ["principio/cita del libro X usado", "..."]
}}"""


# ====================================================================== #
#  AGENTE 3 — Auditor (rigor académico)                                  #
# ====================================================================== #

def build_auditor_prompt(
    question: str, context: str, memory: Dict[str, Any]
) -> str:
    prev_summary = json.dumps(
        {k: memory[k] for k in ["mentor_intake", "investigador"] if k in memory},
        ensure_ascii=False,
    )
    return f"""Eres el AGENTE AUDITOR de rigor académico y calidad científica.

ROL: Auditar la coherencia, consistencia y rigor del fragmento de tesis.

=== PREGUNTA ===
{question}

=== CONTEXTO DE LA TESIS ===
{context}

=== EVALUACIONES PREVIAS ===
{prev_summary}

=== INSTRUCCIONES ===
1. Verifica la coherencia interna del argumento.
2. Detecta inconsistencias, contradicciones o afirmaciones sin soporte.
3. Evalúa el uso correcto de terminología académica.
4. Identifica brechas o vacíos en el desarrollo.
5. Señala los problemas críticos que deben corregirse.

RESPONDE ÚNICAMENTE en formato JSON válido:
{{
  "nivel_rigor": "alto|medio|bajo",
  "coherencia_interna": "alta|media|baja",
  "inconsistencias": ["inconsistencia1"],
  "terminologia": "correcta|parcialmente_correcta|incorrecta",
  "brechas_detectadas": ["brecha1"],
  "problemas_criticos": ["problema1"],
  "puntuacion_rigor": 7.0,
  "recomendaciones": ["recomendacion1", "recomendacion2"]
}}"""


# ====================================================================== #
#  AGENTE 4 — Metodológico (análisis del marco metodológico)             #
# ====================================================================== #

def build_metodologico_prompt(
    question: str,
    context: str,
    memory: Dict[str, Any],
    reference_context: str = "",
) -> str:
    prev_summary = json.dumps(
        {k: memory[k] for k in ["mentor_intake", "investigador", "auditor"] if k in memory},
        ensure_ascii=False,
    )
    refs_block = (
        f"\n=== BIBLIOTECA METODOLÓGICA (libros de referencia) ===\n{reference_context}\n"
        if reference_context else ""
    )
    refs_instr = (
        "6. CRÍTICO: contrasta lo que hace la tesis con las recomendaciones de los libros "
        "de la Biblioteca. Si difieren, indícalo explícitamente. Si coinciden, refuerza la "
        "evaluación citando la fuente. La Biblioteca es tu fuente de verdad metodológica.\n"
        if reference_context else ""
    )
    return f"""Eres el AGENTE METODOLÓGICO especializado en marcos y diseños de investigación científica.

ROL: Evaluar el enfoque y diseño metodológico presente en el fragmento de tesis.

=== PREGUNTA ===
{question}

=== CONTEXTO DE LA TESIS ===
{context}
{refs_block}
=== EVALUACIONES PREVIAS ===
{prev_summary}

=== INSTRUCCIONES ===
1. Identifica el enfoque metodológico (cualitativo/cuantitativo/mixto).
2. Evalúa si el diseño de investigación es adecuado al problema.
3. Analiza instrumentos o técnicas de recolección mencionados.
4. Identifica limitaciones metodológicas explícitas o implícitas.
5. Sugiere ajustes metodológicos concretos.
{refs_instr}
RESPONDE ÚNICAMENTE en formato JSON válido:
{{
  "enfoque": "cualitativo|cuantitativo|mixto|no_especificado",
  "tipo_investigacion": "descriptiva|explicativa|correlacional|experimental|exploratoria|mixta",
  "diseno": "descripción del diseño identificado",
  "adecuacion_metodologica": "alta|media|baja",
  "instrumentos_identificados": ["instrumento1"],
  "limitaciones_metodologicas": ["limitacion1"],
  "sugerencias_metodologicas": ["sugerencia1"],
  "puntuacion_metodologia": 7.0,
  "comentario": "análisis metodológico en 2-3 oraciones",
  "alineacion_con_biblioteca": "alta|media|baja|no_aplica",
  "citas_biblioteca": ["principio metodológico usado del libro X", "..."]
}}"""


# ====================================================================== #
#  AGENTE 5 — Redactor (mejora de escritura académica)                   #
# ====================================================================== #

def build_redactor_prompt(
    question: str, context: str, memory: Dict[str, Any]
) -> str:
    prev_summary = json.dumps(
        {
            k: memory[k]
            for k in ["mentor_intake", "auditor", "metodologico"]
            if k in memory
        },
        ensure_ascii=False,
    )
    return f"""Eres el AGENTE REDACTOR especializado en escritura académica en español.

ROL: Mejorar la calidad de redacción y presentación del fragmento de tesis más relevante.

=== PREGUNTA ===
{question}

=== CONTEXTO DE LA TESIS (fragmento a mejorar) ===
{context[:800]}

=== EVALUACIONES PREVIAS ===
{prev_summary}

=== INSTRUCCIONES ===
1. Selecciona el fragmento más relevante del contexto para mejorar.
2. Reescribe el fragmento con mayor claridad, precisión y estilo académico.
3. Mantén EXACTAMENTE el significado original; solo mejora la forma.
4. Lista los cambios específicos realizados.
5. Provee sugerencias generales de escritura para la tesis completa.

RESPONDE ÚNICAMENTE en formato JSON válido:
{{
  "fragmento_original": "fragmento seleccionado del contexto",
  "fragmento_mejorado": "versión mejorada y clara del fragmento",
  "cambios_realizados": ["cambio1", "cambio2", "cambio3"],
  "nivel_escritura_original": "alto|medio|bajo",
  "sugerencias_generales": ["sugerencia1", "sugerencia2"],
  "comentario": "comentario sobre la calidad de escritura en 2-3 oraciones"
}}"""


# ====================================================================== #
#  AGENTE 6 — Síntesis y Consenso (cierre del panel multiagente)         #
# ====================================================================== #

def build_mentor_final_prompt(
    question: str,
    memory: Dict[str, Any],
    previous_iteration: str | None = None,
) -> str:
    # Serialización compacta (sin indent) para reducir el tamaño del prompt
    # y evitar que el agente 6 reciba >2000 tokens de contexto de agentes previos.
    full_memory = json.dumps(memory, ensure_ascii=False, separators=(",", ":"))

    # Bloque opcional con la síntesis de la iteración anterior. Si está
    # presente, el agente debe refinarla, no repetirla. Si está vacío, el
    # prompt funciona idéntico al original (primera iteración).
    iter_block = (
        f"\n=== SÍNTESIS DE LA ITERACIÓN ANTERIOR ===\n{previous_iteration}\n"
        if previous_iteration else ""
    )
    iter_extra_instr = (
        "10. CRÍTICO: recibiste la SÍNTESIS DE LA ITERACIÓN ANTERIOR. Tu tarea NO es "
        "repetirla — es refinarla. Conserva lo que sigue siendo válido, agudiza lo que "
        "quedó genérico, ajusta puntuación si el panel reveló matices nuevos, y revisita "
        "el debate/consenso/disenso para incorporar precisiones. Cada iteración del panel "
        "debe agregar valor.\n"
        if previous_iteration else ""
    )

    return f"""Eres SÍNTESIS Y CONSENSO, el agente final del panel multiagente de evaluación de tesis.

ROL: Integrar las evaluaciones del Supervisor, Investigador, Auditor, Metodólogo y Redactor en (a) un feedback pedagógico final y (b) la transcripción del DEBATE entre las 3 perspectivas centrales del panel.

Las 3 perspectivas del debate son:
  • Perspectiva FORMAL       → la voz del Auditor (rigor, coherencia, citas).
  • Perspectiva METODOLÓGICA → la voz del Metodólogo (diseño, instrumentos, validez).
  • Perspectiva CONTEXTUAL   → la voz del Investigador (literatura, antecedentes, evidencia).

=== PREGUNTA ORIGINAL ===
{question}

=== EVALUACIONES COMPLETAS DEL PANEL ===
{full_memory}
{iter_block}
=== INSTRUCCIONES ===
1. Sintetiza los hallazgos más importantes de TODOS los agentes previos.
2. Identifica los 3 puntos fuertes principales de la tesis.
3. Lista las 3 áreas de mejora más urgentes.
4. Genera recomendaciones concretas y priorizadas (máximo 5).
5. Calcula la puntuación general (promedio ponderado de las puntuaciones previas).
6. Redacta un mensaje constructivo, motivador y pedagógico para el estudiante.
7. Indica el SIGUIENTE PASO concreto más importante.
8. Reconstruye el DEBATE: resume en 2-3 oraciones lo que dijo cada una de las 3 perspectivas (formal/metodológica/contextual) y produce una síntesis breve.
9. Lista 2-4 puntos de CONSENSO (donde las 3 perspectivas coinciden) y 2-4 puntos de DISENSO (donde 2 perspectivas chocan o se contradicen). Sé honesto: si no hay disenso real, devolvé [].
{iter_extra_instr}

RESPONDE ÚNICAMENTE en formato JSON válido:
{{
  "resumen_ejecutivo": "resumen claro en 3-5 oraciones para el estudiante",
  "puntos_fuertes": ["punto1", "punto2", "punto3"],
  "areas_mejora": ["area1", "area2", "area3"],
  "recomendaciones_priorizadas": [
    {{"prioridad": 1, "recomendacion": "...", "justificacion": "..."}},
    {{"prioridad": 2, "recomendacion": "...", "justificacion": "..."}},
    {{"prioridad": 3, "recomendacion": "...", "justificacion": "..."}}
  ],
  "puntuacion_general": 7.2,
  "nivel_tesis": "excelente|buena|aceptable|necesita_mejoras|insuficiente",
  "mensaje_pedagogico": "mensaje motivador y constructivo para el estudiante",
  "siguiente_paso": "acción concreta más importante que debe realizar ahora",
  "debate": {{
    "perspectiva_formal":       "resumen 2-3 oraciones de lo que dijo el Auditor",
    "perspectiva_metodologica": "resumen 2-3 oraciones de lo que dijo el Metodólogo",
    "perspectiva_contextual":   "resumen 2-3 oraciones de lo que dijo el Investigador",
    "sintesis":                 "síntesis breve de cómo las 3 perspectivas se integran o se tensionan"
  }},
  "consenso": [
    "punto donde las 3 perspectivas coinciden 1",
    "punto donde las 3 perspectivas coinciden 2"
  ],
  "disenso": [
    "punto donde 2 perspectivas chocan o se contradicen 1",
    "(si no hay disenso real, devolver array vacío)"
  ]
}}"""


# ====================================================================== #
#  TEXTO SUGERIDO — Reescritura mejorada de la sección analizada         #
# ====================================================================== #

def build_texto_sugerido_prompt(
    original_context: str,
    question: str,
    final_evaluation: Dict[str, Any],
    investigador_findings: Dict[str, Any],
) -> str:
    """
    Construye el prompt para generar un texto académico mejorado que
    reemplace la sección analizada.  El Investigador es el agente clave:
    sus debilidades y sugerencias guían qué debe cambiar en el contenido.
    La evaluación final aporta las recomendaciones priorizadas y áreas de mejora.
    """
    # ── Datos del Mentor Final (o evaluación Flowise) ─────────────────────
    areas_mejora      = final_evaluation.get("areas_mejora", [])
    puntos_fuertes    = final_evaluation.get("puntos_fuertes", [])
    recomendaciones   = final_evaluation.get("recomendaciones_priorizadas", [])

    # ── Datos del Investigador ────────────────────────────────────────────
    debilidades_inv   = investigador_findings.get("debilidades", [])
    sugerencias_inv   = investigador_findings.get("sugerencias", [])
    respaldo          = investigador_findings.get("respaldo_teorico", "")

    # ── Formateo de secciones opcionales ─────────────────────────────────
    def bullet(items):
        return "\n".join(f"  • {i}" for i in items) if items else "  • (no especificado)"

    recs_text = "\n".join(
        f"  [{r.get('prioridad', i + 1)}] {r.get('recomendacion', str(r))}"
        + (f"\n      → {r['justificacion']}" if r.get("justificacion") else "")
        for i, r in enumerate(recomendaciones[:4])
    ) if recomendaciones else "  [1] Mejorar profundidad argumentativa y respaldo teórico"

    investigador_block = ""
    if debilidades_inv or sugerencias_inv:
        investigador_block = f"""
=== HALLAZGOS DEL AGENTE INVESTIGADOR ===
Debilidades del contenido que DEBES corregir:
{bullet(debilidades_inv)}

Sugerencias de investigación para enriquecer el texto:
{bullet(sugerencias_inv)}
{f"Nivel de respaldo teórico actual: {respaldo}" if respaldo else ""}
"""

    return f"""Eres un EXPERTO EN REDACCIÓN ACADÉMICA universitaria en español.

=== TEXTO ORIGINAL A MEJORAR ===
{original_context}

=== EVALUACIÓN QUE SE USÓ COMO BASE ===
Pregunta evaluada: {question}

Puntos fuertes a CONSERVAR:
{bullet(puntos_fuertes)}

Áreas de mejora a CORREGIR:
{bullet(areas_mejora)}

Recomendaciones priorizadas a IMPLEMENTAR:
{recs_text}
{investigador_block}
=== TU TAREA ===
Reescribe el texto original produciendo una versión mejorada directamente
usable en el documento de tesis.

REGLAS ESTRICTAS:
1. Conserva la misma sección académica y tema del original.
2. Corrige CADA área de mejora listada.
3. Implementa las recomendaciones, en especial las de mayor prioridad.
4. Integra las sugerencias del Investigador para fortalecer el argumento.
5. Mantén y refuerza los puntos fuertes identificados.
6. Usa lenguaje académico formal en español; mejora la cohesión y precisión.
7. La extensión debe ser igual o mayor a la del original.
8. NO inventes datos, estadísticas ni citas que no figuren en el texto original.
9. Devuelve ÚNICAMENTE el texto mejorado: sin títulos, sin explicaciones,
   sin formato markdown, listo para copiar y pegar en la tesis."""
