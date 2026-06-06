"""
Servicio de rúbrica especializada (Hernández-Sampieri 2018).

Responsabilidades:
  - Ingesta de `rubrica.json` (100 pts, 15 secciones, pesos por ítem).
  - Selección DINÁMICA de secciones: cuando se evalúa una PARTE de la tesis
    (p.ej. "Formulación del problema"), se usan únicamente las secciones de la
    rúbrica que correspondan a esa parte. El subconjunto se razona con un LLM
    (con fallback heurístico determinista para tests / fallos de red).
  - Calificación del texto contra las secciones seleccionadas: por cada ítem
    devuelve {id, criterio, pts_max, calificacion, justificacion}; agrega
    subtotales por sección, total, % sobre el máximo de las secciones elegidas,
    y las DOS notas que conviven con G-Eval: Puntaje 0-10 y Nota vigesimal 0-20.

Este módulo NO instancia LLMs: recibe un `llm_invoke` (async callable que toma
un prompt str y devuelve la respuesta str). Así queda desacoplado del proveedor
y es testeable sin coste de tokens. `judge_service` provee ese `llm_invoke`.
"""
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Firma del invocador LLM que inyectan los callers (judge_service).
LLMInvoke = Callable[[str], Awaitable[str]]

# `rubrica.json` vive en la raíz del repo (un nivel arriba de services/).
_RUBRICA_PATH = Path(__file__).resolve().parent.parent / "rubrica.json"

# Sentinela para "evaluación integral" (todas las secciones). Coincide con el
# _OVERVIEW_SECTION_ID del frontend ("__overview__") y con seccion=None.
_OVERVIEW_TOKENS = {"__overview__", "", "vista general", "overview", "general"}


# ---------------------------------------------------------------------- #
#  Ingesta                                                                #
# ---------------------------------------------------------------------- #

_RUBRICA_CACHE: Optional[Dict[str, Any]] = None


def load_rubrica() -> Dict[str, Any]:
    """Carga y cachea `rubrica.json`. Lanza si el archivo no existe o es inválido."""
    global _RUBRICA_CACHE
    if _RUBRICA_CACHE is None:
        with open(_RUBRICA_PATH, encoding="utf-8") as fh:
            _RUBRICA_CACHE = json.load(fh)
        n = len(_RUBRICA_CACHE.get("secciones", []))
        logger.info(f"📋 Rúbrica cargada: {n} secciones, {puntaje_total()} pts máx.")
    return _RUBRICA_CACHE


def all_sections() -> List[Dict[str, Any]]:
    """Lista de las 15 secciones (cada una con numero, nombre, puntaje_maximo, items)."""
    return load_rubrica().get("secciones", [])


def puntaje_total() -> float:
    """Puntaje máximo total declarado en la rúbrica (100)."""
    return float(load_rubrica().get("puntaje_total", 100))


def section_by_number(numero: int) -> Optional[Dict[str, Any]]:
    for sec in all_sections():
        if int(sec.get("numero", -1)) == int(numero):
            return sec
    return None


# ---------------------------------------------------------------------- #
#  Selección dinámica de secciones                                        #
# ---------------------------------------------------------------------- #

def _is_overview(seccion_label: Optional[str]) -> bool:
    if seccion_label is None:
        return True
    return seccion_label.strip().lower() in _OVERVIEW_TOKENS


def _norm(text: str) -> str:
    """Minúsculas sin acentos (NFKD) para comparar nombres de sección de forma robusta."""
    nfkd = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# Palabras vacías o demasiado comunes en títulos académicos: no discriminan sección.
_STOP_WORDS = {
    "de", "del", "la", "el", "los", "las", "y", "e", "en", "un", "una",
    "estudio", "proyecto", "tesis", "investigacion", "datos",
}


def _significant_words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"\w+", _norm(text))
        if w not in _STOP_WORDS and len(w) > 3
    }


def _heuristic_select(seccion_label: str) -> List[int]:
    """
    Fallback determinista cuando no hay LLM o falla: empareja por número de
    prefijo (p.ej. "3 Formulación…" → sección 3) o por solapamiento de palabras
    significativas (sin acentos) con el nombre. Si nada coincide, devuelve []
    (el caller cae a todas).
    """
    # 1) Prefijo numérico: "3.2 Algo" / "3 - Algo" → 3
    m = re.match(r"^\s*(\d+)", seccion_label.strip())
    if m:
        num = int(m.group(1))
        if section_by_number(num):
            return [num]

    # 2) Solapamiento de palabras significativas con el nombre de la sección.
    label_words = _significant_words(seccion_label)
    best_num, best_overlap = None, 0
    for sec in all_sections():
        overlap = len(label_words & _significant_words(sec.get("nombre", "")))
        if overlap > best_overlap:
            best_num, best_overlap = int(sec["numero"]), overlap
    return [best_num] if best_num and best_overlap >= 1 else []


def _build_select_prompt(seccion_label: str) -> str:
    catalogo = "\n".join(
        f"  {sec['numero']}. {sec['nombre']} (máx {sec['puntaje_maximo']} pts)"
        for sec in all_sections()
    )
    return (
        "Eres un metodólogo experto en proyectos de tesis (Hernández-Sampieri 2018).\n"
        "Tienes el catálogo de secciones de una rúbrica de evaluación:\n\n"
        f"{catalogo}\n\n"
        f"Se va a evaluar ÚNICAMENTE esta parte del proyecto: \"{seccion_label}\".\n\n"
        "Razona qué secciones de la rúbrica aplican DIRECTAMENTE a esa parte y "
        "cuáles NO (no incluyas secciones de otras partes como referencias, marco "
        "teórico o instrumentos si la parte no las trata).\n"
        "Responde ÚNICAMENTE con un JSON válido de esta forma exacta:\n"
        '{\"secciones\": [<números enteros>]}'
    )


async def select_sections(
    seccion_label: Optional[str],
    *,
    llm_invoke: Optional[LLMInvoke] = None,
) -> List[Dict[str, Any]]:
    """
    Devuelve la LISTA de secciones de la rúbrica aplicables a la parte indicada.

    - Vista general / None → las 15 secciones.
    - Parte concreta → subconjunto razonado por el LLM (`llm_invoke`). Si no se
      pasa LLM o falla, usa la heurística; si la heurística tampoco resuelve,
      cae a todas (conservador: nunca deja al usuario sin evaluación).
    """
    if _is_overview(seccion_label):
        return all_sections()

    label = seccion_label.strip()
    numeros: List[int] = []

    if llm_invoke is not None:
        try:
            raw = await llm_invoke(_build_select_prompt(label))
            data = _parse_json(raw)
            numeros = [int(n) for n in data.get("secciones", []) if section_by_number(int(n))]
        except Exception as exc:
            logger.warning(f"select_sections: LLM falló ({exc}); usando heurística.")

    if not numeros:
        numeros = _heuristic_select(label)

    if not numeros:
        logger.info(
            f"select_sections('{label}'): sin match LLM/heurística; "
            "usando TODAS las secciones."
        )
        return all_sections()

    seleccion = [section_by_number(n) for n in sorted(set(numeros))]
    seleccion = [s for s in seleccion if s]
    logger.info(
        f"select_sections('{label}') → secciones {[s['numero'] for s in seleccion]}"
    )
    return seleccion


# ---------------------------------------------------------------------- #
#  Calificación contra la rúbrica                                         #
# ---------------------------------------------------------------------- #

def _parse_json(text: str) -> Dict[str, Any]:
    """Extrae JSON de la respuesta LLM (tolerante a markdown / texto alrededor)."""
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _build_score_prompt(text: str, secciones: List[Dict[str, Any]]) -> str:
    items_catalog: List[str] = []
    for sec in secciones:
        items_catalog.append(f"\nSección {sec['numero']} — {sec['nombre']}:")
        for it in sec.get("items", []):
            items_catalog.append(
                f"  [{it['id']}] (máx {it['pts_max']} pts) {it['criterio']}"
            )
    catalogo = "\n".join(items_catalog)
    return (
        "Eres un evaluador metodológico experto (Hernández-Sampieri 2018) y debes "
        "calificar el siguiente texto de un proyecto de tesis SOLO contra los ítems "
        "de rúbrica indicados.\n\n"
        "REGLA DE PUNTUACIÓN por ítem: asigna el puntaje MÁXIMO (pts_max) si el "
        "criterio SE CUMPLE COMPLETAMENTE, el 50% si se cumple parcialmente, o 0 si "
        "no se cumple.\n\n"
        f"=== ÍTEMS A CALIFICAR ===\n{catalogo}\n\n"
        f"=== TEXTO A EVALUAR ===\n{text}\n\n"
        "Responde ÚNICAMENTE con JSON válido de esta forma EXACTA:\n"
        '{\"items\": [{\"id\": \"3.1\", \"calificacion\": <número>, '
        '\"justificacion\": \"<1-2 frases>\"}, ...]}\n'
        "Incluye TODOS los ítems listados. `calificacion` debe ser pts_max, su 50%, o 0."
    )


def _clamp_calificacion(valor: Any, pts_max: float) -> float:
    """Acota la calificación del LLM a [0, pts_max] y la snap-ea a {0, 50%, max}."""
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return 0.0
    v = max(0.0, min(v, pts_max))
    # Snap al valor de rúbrica más cercano (0 / parcial / completo).
    candidatos = [0.0, round(pts_max / 2, 2), float(pts_max)]
    return min(candidatos, key=lambda c: abs(c - v))


def aggregate_scores(
    secciones: List[Dict[str, Any]],
    calificaciones: Dict[str, float],
) -> Dict[str, Any]:
    """
    Agrega calificaciones por ítem (dict id→puntos) en subtotales por sección,
    total, máximo de las secciones elegidas, %, Puntaje 0-10 y Nota vigesimal.

    Es determinista (sin LLM) → unidad testeable por separado.
    """
    secciones_out: List[Dict[str, Any]] = []
    total_obtenido = 0.0
    total_maximo = 0.0

    for sec in secciones:
        items_out: List[Dict[str, Any]] = []
        subtotal = 0.0
        for it in sec.get("items", []):
            pts_max = float(it["pts_max"])
            obtenido = _clamp_calificacion(calificaciones.get(it["id"], 0.0), pts_max)
            subtotal += obtenido
            items_out.append({
                "id": it["id"],
                "criterio": it["criterio"],
                "pts_max": pts_max,
                "calificacion": obtenido,
            })
        sec_max = float(sec.get("puntaje_maximo", sum(float(i["pts_max"]) for i in sec.get("items", []))))
        total_obtenido += subtotal
        total_maximo += sec_max
        secciones_out.append({
            "numero": sec["numero"],
            "nombre": sec["nombre"],
            "subtotal": round(subtotal, 2),
            "subtotal_maximo": sec_max,
            "items": items_out,
        })

    pct = (total_obtenido / total_maximo) if total_maximo > 0 else 0.0
    return {
        "secciones": secciones_out,
        "total_obtenido": round(total_obtenido, 2),
        "total_maximo": round(total_maximo, 2),
        "porcentaje": round(pct, 4),
        "puntaje_0_10": round(pct * 10, 2),
        "nota_vigesimal": round(pct * 20, 2),
    }


async def score_text(
    text: str,
    secciones: List[Dict[str, Any]],
    *,
    llm_invoke: LLMInvoke,
) -> Dict[str, Any]:
    """
    Califica `text` contra las `secciones` seleccionadas usando `llm_invoke`.

    Devuelve el dict de `aggregate_scores` enriquecido con la `justificacion`
    de cada ítem (la nota numérica se snap-ea a los valores válidos de rúbrica;
    la justificación se conserva tal cual la dio el juez).
    """
    raw = await llm_invoke(_build_score_prompt(text, secciones))
    data = _parse_json(raw)

    califs: Dict[str, float] = {}
    justifs: Dict[str, str] = {}
    for entry in data.get("items", []):
        iid = str(entry.get("id", "")).strip()
        if not iid:
            continue
        califs[iid] = entry.get("calificacion", 0.0)
        justifs[iid] = str(entry.get("justificacion", "")).strip()

    agg = aggregate_scores(secciones, califs)
    for sec in agg["secciones"]:
        for it in sec["items"]:
            it["justificacion"] = justifs.get(it["id"], "")
    return agg
