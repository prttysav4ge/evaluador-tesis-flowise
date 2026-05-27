"""
🎓 Evaluador de Proyecto de Investigación — Interfaz Streamlit (TODO EN UNO)
==========================================
Interfaz visual de 3 pantallas para el POC RAG Multiagente:

  📄  Cargar PDF      → sube y procesa el PDF del proyecto de investigación
  🔬  Ver Embeddings  → visualiza cómo el PDF se fragmentó y almacenó
  💬  Consultar       → envía preguntas a los agentes Flowise / Python

Esta versión ejecuta el backend FastAPI EN EL MISMO PROCESO usando
`FastAPI.TestClient`. No requiere `python main.py` por separado: ideal
para Streamlit Cloud donde no hay manera de levantar un segundo servicio.
"""

# ─────────────────────────────────────────────
#  BOOTSTRAP: inyectar st.secrets en os.environ
#  (DEBE ir antes de importar app.config o cualquier módulo del backend)
# ─────────────────────────────────────────────
import os
import streamlit as st

try:
    _secrets = dict(st.secrets)
    for _k, _v in _secrets.items():
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))
except Exception:
    # Sin secrets.toml ni Secrets en Streamlit Cloud → caemos al .env local
    pass

import time
import uuid
from typing import Any, Dict
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────
#  BACKEND EN-PROCESO (FastAPI vía TestClient)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Iniciando backend en-proceso (puede tardar ~30s la primera vez: descarga del modelo de embeddings)…")
def get_backend_client() -> TestClient:
    """
    Levanta la app FastAPI dentro del proceso de Streamlit y devuelve un
    TestClient. El `__enter__` dispara el lifespan (inicializa ChromaDB y
    descarga/carga el modelo `multilingual-e5-small`).

    @st.cache_resource garantiza que esto solo se ejecute una vez por sesión.
    """
    from main import app
    client = TestClient(app)
    client.__enter__()  # dispara lifespan (init ChromaDB + embeddings)
    return client

SECTION_COLORS = {
    "resumen":              "#4CAF50",
    "introduccion":         "#2196F3",
    "planteamiento_problema":"#F44336",
    "justificacion":        "#FF9800",
    "objetivos":            "#9C27B0",
    "hipotesis":            "#E91E63",
    "antecedentes":         "#00BCD4",
    "estado_del_arte":      "#009688",
    "marco_teorico":        "#3F51B5",
    "marco_conceptual":     "#673AB7",
    "marco_metodologico":   "#795548",
    "metodologia":          "#607D8B",
    "diseno_investigacion": "#FF5722",
    "resultados":           "#8BC34A",
    "analisis":             "#FFC107",
    "discusion":            "#03A9F4",
    "conclusiones":         "#4DB6AC",
    "referencias":          "#90A4AE",
    "general":              "#BDBDBD",
}

SECTION_LABELS = {
    "resumen":               "Resumen",
    "introduccion":          "Introducción",
    "planteamiento_problema":"Planteamiento del Problema",
    "justificacion":         "Justificación",
    "objetivos":             "Objetivos",
    "hipotesis":             "Hipótesis",
    "antecedentes":          "Antecedentes",
    "estado_del_arte":       "Estado del Arte",
    "marco_teorico":         "Marco Teórico",
    "marco_conceptual":      "Marco Conceptual",
    "marco_metodologico":    "Marco Metodológico",
    "metodologia":           "Metodología",
    "diseno_investigacion":  "Diseño de Investigación",
    "resultados":            "Resultados",
    "analisis":              "Análisis",
    "discusion":             "Discusión",
    "conclusiones":          "Conclusiones",
    "referencias":           "Referencias",
    "general":               "General / Sin clasificar",
}


# ─────────────────────────────────────────────
#  ESTADO DE LA APLICACIÓN (workflow + session)
# ─────────────────────────────────────────────
# Etapas del workflow — driven por st.session_state["workflow_stage"].
# La pantalla principal se elige según esta clave (ver Sprint 1 commit 3).
STAGE_UPLOAD     = "upload"      # sin PDF cargado
STAGE_CONFIGURE  = "configure"   # PDF vectorizado, eligiendo sección
STAGE_RESULTS    = "results"     # evaluación completada, mostrando 4 pestañas
STAGE_EMBEDDINGS = "embeddings"  # vista de fragmentación (acceso opcional)

# Rúbricas disponibles. Por ahora solo UPAO; el dropdown del sidebar la elige.
# El número de ítems es placeholder hasta que se concrete la rúbrica real (Sprint 3).
RUBRICS = {
    "upao_ing_sistemas": {
        "label":   "UPAO · Ing. Sistemas",
        "items":   33,
        "version": "oficial",
    },
}

# Keys de st.session_state agrupadas por scope de reset.
_SESSION_KEYS_PDF = (
    "pdf_uploaded",
    "pdf_filename",
    "pdf_sections",
    "pdf_outline",
    "pdf_chunks_total",
    "custom_rubric_filename",   # nombre del PDF de rúbrica subido (sin procesar todavía)
)
_SESSION_KEYS_RESULT = (
    "last_result",
    "last_question",
    "last_metrics",   # cache de métricas NLP (calculadas on-demand en Pestaña 4)
)
_SESSION_KEYS_CONFIG = (
    "thread_id",
    "rubric_id",
    "iterations",
    "selected_section_id",
    "workflow_stage",
)


def init_session_state() -> None:
    """
    Inicializa todas las keys de st.session_state con sus defaults.
    Idempotente: setdefault no sobrescribe valores ya seteados, así que
    se puede llamar al inicio de main() en cada rerun sin perder estado.
    """
    defaults = {
        # config
        "thread_id":             str(uuid.uuid4()),
        "rubric_id":             "upao_ing_sistemas",
        "iterations":             2,
        "selected_section_id":   "__overview__",   # dropdown — "Vista general" por defecto
        "workflow_stage":        STAGE_UPLOAD,
        # pdf
        "pdf_uploaded":          False,
        "pdf_filename":          "",
        "pdf_sections":          {},     # legacy keyword-based dict
        "pdf_outline":           [],     # outline jerárquico (1.1.1) — alimentado por /upload-pdf
        "pdf_chunks_total":      0,
        "custom_rubric_filename": "",     # Paso 2: PDF de rúbrica opcional (solo nombre, no procesado)
        # result
        "last_result":     None,
        "last_question":   "",
        # historial (preexistente — preservado por compatibilidad)
        "query_history":   [],
        # Biblioteca Metodológica — alimentado por Phase 6 con datos reales
        # del endpoint /api/v1/reference-books. None hasta primer fetch.
        "reference_books": None,
    }
    for key, default in defaults.items():
        st.session_state.setdefault(key, default)


def reset_all_state() -> None:
    """
    Reset completo: PDF, configuración, resultados, historial.
    Llamado por el botón 'Nueva evaluación' del sidebar.
    Genera un nuevo thread_id.
    """
    for key in (*_SESSION_KEYS_PDF, *_SESSION_KEYS_RESULT, *_SESSION_KEYS_CONFIG):
        st.session_state.pop(key, None)
    st.session_state["query_history"] = []
    init_session_state()


def reset_for_new_section() -> None:
    """
    Reset parcial: conserva el PDF vectorizado y el thread_id; sólo limpia
    el resultado para que el usuario pueda elegir otra sección sin re-subir.
    """
    for key in _SESSION_KEYS_RESULT:
        st.session_state.pop(key, None)
    st.session_state["workflow_stage"] = STAGE_CONFIGURE
    init_session_state()


def mark_pdf_uploaded(
    filename: str,
    sections: dict,
    chunks_total: int,
    outline: list | None = None,
) -> None:
    """
    Marca el PDF como vectorizado y avanza el workflow al stage 'configure'.
    Llamado por page_upload() tras un upload exitoso.

    Args:
        filename:     nombre original del PDF.
        sections:     dict keyword-based (legacy, conteo por categoría).
        chunks_total: total de chunks almacenados en ChromaDB.
        outline:      lista de encabezados jerárquicos (1.1.1) con
                      chunks_count y chars_count. Vacía si el PDF no
                      usa numeración (el frontend cae a sections).
    """
    st.session_state["pdf_uploaded"]     = True
    st.session_state["pdf_filename"]     = filename
    st.session_state["pdf_sections"]     = sections or {}
    st.session_state["pdf_outline"]      = outline or []
    st.session_state["pdf_chunks_total"] = chunks_total
    st.session_state["workflow_stage"]   = STAGE_CONFIGURE


def thread_id_short(thread_id: str | None = None) -> str:
    """Devuelve la versión truncada del thread_id (ej. '5eb0144e-80b…')."""
    tid = thread_id or st.session_state.get("thread_id", "")
    if not tid:
        return "—"
    return f"{tid[:12]}…"


def workflow_stage_badge() -> tuple[str, str]:
    """Devuelve (texto, emoji) del badge según el stage actual."""
    stage = st.session_state.get("workflow_stage", STAGE_UPLOAD)
    if stage == STAGE_UPLOAD:
        return "Sin PDF cargado", "🟠"
    if stage == STAGE_CONFIGURE:
        return "PDF listo — elige sección", "🔵"
    if stage == STAGE_RESULTS:
        return "Proceso completado", "🟢"
    if stage == STAGE_EMBEDDINGS:
        return "Visualizando fragmentos", "🔵"
    return stage, "⚪"


# ─────────────────────────────────────────────
#  HELPERS DE API (llaman al backend en-proceso via TestClient)
# ─────────────────────────────────────────────
API_PREFIX = "/api/v1"


def _client() -> TestClient:
    return get_backend_client()


def api_health():
    try:
        r = _client().get(f"{API_PREFIX}/health")
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_collection_info():
    try:
        r = _client().get(f"{API_PREFIX}/collection")
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_upload_pdf(file_bytes, filename):
    try:
        r = _client().post(
            f"{API_PREFIX}/upload-pdf",
            files={"file": (filename, file_bytes, "application/pdf")},
        )
        return r.json(), r.status_code
    except Exception as e:
        return {"detail": str(e)}, 500


def api_list_chunks(limit=50, offset=0):
    try:
        r = _client().get(f"{API_PREFIX}/chunks", params={"limit": limit, "offset": offset})
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def api_query(question, top_k=5, session_id=None):
    payload = {"question": question, "top_k": top_k}
    if session_id:
        payload["session_id"] = session_id
    try:
        # El TestClient ejecuta el handler en el mismo proceso (sin red), así que
        # no hay timeout de socket: bloquea hasta que el pipeline termine.
        r = _client().post(f"{API_PREFIX}/query", json=payload)
        return r.json(), r.status_code
    except Exception as e:
        return {"detail": str(e)}, 500


def api_reset_collection():
    try:
        r = _client().delete(f"{API_PREFIX}/collection", params={"confirm": "true"})
        return r.json(), r.status_code
    except Exception as e:
        return {"detail": str(e)}, 500


# ─────────────────────────────────────────────
#  COMPONENTES REUTILIZABLES
# ─────────────────────────────────────────────
def render_health_badge():
    """Muestra estado del sistema en la barra lateral (versión verbose, legacy).

    DEPRECATED: reemplazada por render_health_badge_compact() dentro de
    render_sidebar(). Se conserva por compatibilidad y para no romper
    referencias durante el refactor del Sprint 1.
    """
    health = api_health()
    if health is None:
        st.sidebar.error("⛔ Backend no inicializado")
        st.sidebar.caption("Recarga la página; si persiste, revisa los Secrets en Streamlit Cloud.")
        return False

    comps = health.get("components", {})
    chroma_chunks = comps.get("chromadb", {}).get("chunks_stored", 0)
    flowise_ok = comps.get("flowise", {}).get("reachable", False)
    mode = health.get("execution_mode", "?")

    st.sidebar.success("✅ Backend conectado")
    st.sidebar.metric("Chunks en ChromaDB", chroma_chunks)

    flowise_status = "🟢 Activo" if flowise_ok else "🔴 Sin conexión"
    st.sidebar.caption(f"Flowise: {flowise_status}")
    st.sidebar.caption(f"Modo: **{mode}**")
    return True


def render_health_badge_compact() -> bool:
    """Versión compacta del health badge — vive dentro de render_sidebar()."""
    health = api_health()
    if health is None:
        st.error("⛔ Backend no inicializado")
        st.caption("Revisa los Secrets en Streamlit Cloud.")
        return False
    comps = health.get("components", {})
    flowise_ok = comps.get("flowise", {}).get("reachable", False)
    mode = health.get("execution_mode", "?")
    st.success(f"✅ Backend OK · `{mode}`")
    st.caption(f"Flowise: {'🟢 activo' if flowise_ok else '🔴 sin conexión'}")
    return True


# Lista placeholder de los 4 PDFs metodológicos de referencia.
# Phase 6 reemplaza esto con datos reales leídos desde el endpoint
# /api/v1/reference-books (que indexa los PDFs en una colección Chroma aparte).
_REFERENCE_BOOKS_PLACEHOLDER = [
    {"title": "METODOLOGIA DE LA INVESTIGACION CUANTITATIVA-CUALITATIVA Y REDACCION DE LA TESIS", "fragments": None},
    {"title": "METODOLOGIA DE LA INVESTIGACION-GUIA PARA EL PROYECTO DE TESIS",                   "fragments": None},
    {"title": "METODOLOGÍA DE LA INVESTIGACION-LAS RUTAS CUANTITATIVA, CUALITATIVA Y MIXTA",      "fragments": None},
    {"title": "PROYECTO DE TESIS-GUIA PRACTICA PARA INVESTIGACION CUANTITATIVA",                  "fragments": None},
]


def render_sidebar() -> bool:
    """
    Sidebar persistente alineado con la app de referencia 'Mentoría UPAO':
    header + caption del stack, badge de estado del workflow, botón único de
    reset, selector de rúbrica y biblioteca metodológica.

    Returns:
        bool — True si el backend está inicializado.
    """
    # Backend health lo verificamos pero NO lo mostramos a menos que falle
    # (la referencia no tiene este indicador visible cuando todo va bien).
    health = api_health()
    backend_ok = health is not None

    with st.sidebar:
        # ── Header ────────────────────────────────────────────────────────
        st.title("🌱 Mentoría UPAO")
        st.caption("PoC · Flowise + RAG + Groq Llama 3.3")
        st.markdown("")  # respiro vertical

        # ── Estado del workflow ───────────────────────────────────────────
        if not backend_ok:
            st.error("⛔ Backend no inicializado")
        else:
            badge_text, badge_emoji = workflow_stage_badge()
            st.markdown(f"**Estado:** {badge_emoji} {badge_text}")
        st.markdown("")

        # ── Botón único de reset ─────────────────────────────────────────
        if st.button(
            "🔄 Nueva evaluación",
            use_container_width=True,
            help="Vacía ChromaDB, resetea configuración y resultados, y genera nuevo thread_id.",
        ):
            api_reset_collection()
            reset_all_state()
            st.rerun()

        # ── Universidad / Rúbrica ────────────────────────────────────────
        rubric_keys   = list(RUBRICS.keys())
        rubric_labels = [RUBRICS[k]["label"] for k in rubric_keys]
        current_idx   = rubric_keys.index(
            st.session_state.get("rubric_id", rubric_keys[0])
        )
        chosen_label = st.selectbox(
            "Universidad / Rúbrica:",
            options=rubric_labels,
            index=current_idx,
            key="_sidebar_rubric_select",
            help=(
                "Plantilla de rúbrica oficial. Podés subir tu propia rúbrica de "
                "evaluación en el Paso 2 para personalizar el análisis."
            ),
        )
        st.session_state["rubric_id"] = rubric_keys[
            rubric_labels.index(chosen_label)
        ]
        st.markdown("")

        # ── Biblioteca Metodológica ──────────────────────────────────────
        books = st.session_state.get("reference_books") or _REFERENCE_BOOKS_PLACEHOLDER
        total_fragments = sum(
            b.get("fragments") or 0 for b in books
        )
        st.markdown("### 📚 Biblioteca Metodológica")
        if total_fragments:
            st.caption(f"**{len(books)} libro(s) · {total_fragments:,} fragmentos indexados**")
        else:
            st.caption(f"**{len(books)} libro(s) · indexación pendiente**")

        for book in books:
            frags = book.get("fragments")
            frag_caption = (
                f"{frags:,} fragmentos" if frags
                else "_pendiente de indexación_"
            )
            st.markdown(
                f"📖 **{book['title']}**  \n"
                f"<span style='font-size:0.85em;color:#888'>{frag_caption}</span>",
                unsafe_allow_html=True,
            )

    return backend_ok


def section_badge(section_key: str) -> str:
    label = SECTION_LABELS.get(section_key, section_key)
    color = SECTION_COLORS.get(section_key, "#BDBDBD")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:0.75em;font-weight:600">{label}</span>'


# ─────────────────────────────────────────────
#  PANTALLA 1 — CARGAR PDF
# ─────────────────────────────────────────────
# Umbral debajo del cual una sección se marca con ⚠️ en la tabla
# (señala que probablemente quedó incompleta al detectar el heading).
_FRAGMENT_WARNING_CHARS = 200


def _render_fragmentation_table(
    outline: list,
    sections_found: dict,
    total_chars_fallback: int,
) -> None:
    """
    Renderiza la tabla expandible `Sección | Pág. | Chars | Frags` con
    ⚠️ amarillo para secciones con chars < _FRAGMENT_WARNING_CHARS.

    Prefiere el outline jerárquico (1.1.1) si está disponible; si no, cae
    a sections_found (keyword-based) sin info de página/chars.
    """
    if outline:
        rows = [
            {
                "Sección": (
                    f"⚠️ {h['section_id']} {h['title']}"
                    if h.get("chars_count", 0) < _FRAGMENT_WARNING_CHARS
                    else f"{h['section_id']} {h['title']}"
                ),
                "Pág.":   h["page"],
                "Chars":  h["chars_count"],
                "Frags":  h["chunks_count"],
            }
            for h in outline
        ]
        total_sections = len(outline)
        total_frags    = sum(h["chunks_count"] for h in outline)
        total_chars    = sum(h["chars_count"]  for h in outline)
        source_note    = ""

    elif sections_found:
        # Fallback: sin info de página/chars por sección, solo conteo de chunks.
        # Cuando llega Sprint 3, este caso debería ser raro (la mayoría de
        # tesis tiene numeración 1.1.1).
        rows = [
            {
                "Sección": SECTION_LABELS.get(k, k),
                "Pág.":   "—",
                "Chars":  "—",
                "Frags":  v,
            }
            for k, v in sorted(sections_found.items(), key=lambda x: -x[1])
        ]
        total_sections = len(sections_found)
        total_frags    = sum(sections_found.values())
        total_chars    = total_chars_fallback
        source_note    = (
            " — _(detección por keyword; no se encontró numeración jerárquica `1.1.1`)_"
        )

    else:
        st.info("No se detectaron secciones en el PDF.")
        return

    summary = (
        f"**Fragmentación completada:** {total_sections} secciones · "
        f"{total_frags} fragmentos · {total_chars:,} caracteres totales"
        f"{source_note}"
    )

    with st.expander(summary, expanded=True):
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Sección": st.column_config.TextColumn("Sección", width="large"),
                "Pág.":    st.column_config.Column("Pág.",  width="small"),
                "Chars":   st.column_config.Column("Chars", width="small"),
                "Frags":   st.column_config.Column("Frags", width="small"),
            },
        )
        if any("⚠️" in row["Sección"] for row in rows):
            st.caption(
                f"⚠️ secciones con menos de {_FRAGMENT_WARNING_CHARS} caracteres "
                "(probablemente quedaron incompletas o son sólo títulos sin cuerpo)."
            )


def _render_rubrica_step() -> None:
    """
    Paso 2 — Rúbrica de evaluación (opcional).

    El usuario puede subir un PDF de su rúbrica personalizada. Por ahora
    el archivo NO se procesa (solo se guarda el nombre en session_state).
    Sprint 4 lo vectorizará y lo inyectará como contexto adicional.
    """
    rubric = RUBRICS.get(st.session_state.get("rubric_id", "upao_ing_sistemas"), {})
    rubric_label = rubric.get("label",   "—")
    rubric_items = rubric.get("items",   0)

    st.header("Paso 2 — Rúbrica de evaluación (opcional)")

    col_upload, col_info = st.columns([1, 2])

    with col_upload:
        rubric_file = st.file_uploader(
            "Sube la rúbrica de evaluación (PDF)",
            type=["pdf"],
            help="Opcional. Si no subís nada, se usa la rúbrica oficial UPAO.",
            key="_rubric_uploader",
        )
        if rubric_file is not None:
            st.session_state["custom_rubric_filename"] = rubric_file.name

    with col_info:
        custom = st.session_state.get("custom_rubric_filename", "")
        if custom:
            st.success(
                f"✅ Rúbrica personalizada cargada: **{custom}**\n\n"
                "_Por ahora solo se registra el archivo. La integración "
                "completa con el pipeline llega en una iteración siguiente._"
            )
        else:
            st.info(
                f"Sin rúbrica subida — se usará la **rúbrica oficial {rubric_label}** "
                f"({rubric_items} ítems).  \n"
                "Puedes subir la rúbrica de tu jurado evaluador para obtener "
                "una evaluación personalizada."
            )


def _render_landing_intro() -> None:
    """
    Header + explicación '¿Cómo funciona este sistema?' al estilo de la
    app de referencia. Se muestra siempre al tope de la página de carga
    (incluso después de vectorizar) para mantener la identidad visual.
    """
    st.title("Sistema de Mentoría Académica Multiagente")

    st.markdown("**¿Cómo funciona este sistema?**")
    st.markdown(
        "1. **Sube el PDF** de tu proyecto de tesis borrador\n"
        "2. (Opcional) **Sube tu rúbrica de evaluación** — si no la subes, se usa la rúbrica UPAO por defecto\n"
        "3. **El sistema vectoriza** el documento (embeddings locales, sin enviar datos al exterior)\n"
        "4. **Elige una sección** y el sistema recupera solo ese fragmento (anti-token-burn)\n"
        "5. **Red multiagente** Redactor ↔ Auditor ↔ Metodólogo mejora el texto iterativamente\n"
        "6. **Tú revisas y apruebas** la versión final como mentor"
    )
    st.markdown("---")


def page_upload():
    _render_landing_intro()

    st.header("Paso 1 — Carga el PDF de tu proyecto de tesis")

    # ── Uploader ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Sube el borrador del proyecto de tesis (PDF)",
        type=["pdf"],
        help="El PDF se procesa localmente. Los embeddings se generan en tu máquina.",
    )

    if uploaded is not None:
        # Layout 2 columnas: info banner a la izquierda, Vectorizar +
        # panel de progreso a la derecha.
        col_info, col_action = st.columns([2, 1])

        with col_info:
            st.info(
                f"📄 **{uploaded.name}** ({uploaded.size / 1024:.1f} KB)\n\n"
                "Primera vectorización descarga el modelo "
                "`multilingual-e5-small` (~117 MB). Las siguientes son instantáneas."
            )

        with col_action:
            do_vectorize = st.button(
                "🚀 Vectorizar PDF",
                type="primary",
                use_container_width=True,
            )

        if do_vectorize:
            # Panel de progreso a la derecha mientras corre el pipeline.
            with col_action:
                with st.status("⏳ Procesando PDF de tesis…", expanded=True) as status_box:
                    st.write("Analizando estructura del PDF (separando índice del contenido)…")
                    t0 = time.time()
                    result, status_code = api_upload_pdf(uploaded.read(), uploaded.name)
                    elapsed = round(time.time() - t0, 1)

                    if status_code == 200 and result.get("success"):
                        outline = result.get("outline", []) or []
                        total_pages = result.get("total_pages", 0)
                        total_chars = sum(h.get("chars_count", 0) for h in outline) or \
                                      result.get("chunks_generated", 0) * 800

                        st.write(
                            f"Texto de contenido extraído: **{total_chars:,} caracteres** "
                            f"en **{total_pages} páginas**"
                        )
                        st.write(f"Estructura detectada: **{len(outline)} secciones** en el índice")
                        # Muestra primeros 8 headings + '…' si hay más
                        preview_headings = outline[:8]
                        for h in preview_headings:
                            st.markdown(f"- **{h['section_id']}.** {h['title']}")
                        if len(outline) > len(preview_headings):
                            st.markdown("- …")
                        st.write("Dividiendo contenido por secciones del índice (chunking semántico)…")
                        st.write("Generando embeddings locales (`multilingual-e5-small`)…")

                        status_box.update(
                            label=f"✅ Vectorizado en {elapsed} s",
                            state="complete",
                            expanded=False,
                        )
                    else:
                        status_box.update(
                            label=f"❌ Error ({status_code})",
                            state="error",
                            expanded=True,
                        )
                        st.error(result.get("detail", "Error desconocido"))

            # Render final fuera de las columnas (ancho completo)
            if status_code == 200 and result.get("success"):
                # Marca PDF como vectorizado y avanza workflow → STAGE_CONFIGURE.
                mark_pdf_uploaded(
                    filename=uploaded.name,
                    sections=result.get("sections_found", {}),
                    chunks_total=result.get("chunks_stored", 0),
                    outline=result.get("outline", []),
                )

                # ── Mensaje verde personalizado (formato referencia) ────
                st.success(
                    f"PDF `'{uploaded.name}'` ya está vectorizado."
                )

                # ── Tabla de fragmentación (Sección | Pág. | Chars | Frags)
                _render_fragmentation_table(
                    outline=result.get("outline", []),
                    sections_found=result.get("sections_found", {}),
                    total_chars_fallback=result.get("chunks_generated", 0) * 800,
                )

                st.markdown("---")

                # ── Paso 2 — Rúbrica de evaluación (opcional) ───────────
                _render_rubrica_step()

                st.markdown("")

                # ── Botón único para avanzar al Paso 3 (selección sección)
                if st.button(
                    "Continuar a selección de sección →",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state["workflow_stage"] = STAGE_CONFIGURE
                    st.rerun()
    # Nota: la 'Zona peligrosa - Reiniciar ChromaDB' se eliminó porque
    # 'Nueva evaluación' del sidebar ahora vacía ChromaDB automáticamente.


# ─────────────────────────────────────────────
#  PANTALLA 2 — VER EMBEDDINGS
# ─────────────────────────────────────────────
def page_embeddings():
    # Botón de regreso al stage previo (configure si hay PDF, upload si no).
    if st.button("← Volver", help="Regresa a la pantalla principal."):
        st.session_state["workflow_stage"] = (
            STAGE_CONFIGURE if st.session_state.get("pdf_uploaded") else STAGE_UPLOAD
        )
        st.rerun()

    st.header("🔬 Visualizar Embeddings y Chunks")
    st.markdown(
        "Explora cómo el PDF fue fragmentado y cómo está representado en **ChromaDB**. "
        "Cada *chunk* es un fragmento de texto convertido en un vector de alta dimensión "
        "que permite la búsqueda semántica."
    )

    col_info = api_collection_info()
    if col_info is None:
        st.error("No se puede conectar con el backend. Verifica que FastAPI esté corriendo.")
        return

    total = col_info.get("total_chunks", 0)
    if total == 0:
        st.warning("⚠️ No hay chunks almacenados. Sube primero un PDF en **📄 Cargar PDF**.")
        return

    # ── KPIs de colección ────────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Total de chunks", total)
    k2.metric("Colección", col_info.get("collection", "—"))
    k3.metric("Directorio ChromaDB", col_info.get("persist_dir", "—"))

    # ── Cargar muestra de chunks ─────────────────────────────────────────
    limit = st.slider("Chunks a cargar para análisis", 10, 100, 50, 10)
    chunks_data = api_list_chunks(limit=limit, offset=0)
    if chunks_data is None:
        st.error("Error al obtener chunks del backend.")
        return

    chunks = chunks_data.get("chunks", [])
    if not chunks:
        st.warning("No se pudieron obtener chunks.")
        return

    # Construir dataframe
    rows = []
    for i, c in enumerate(chunks):
        meta = c.get("metadata", {})
        rows.append({
            "idx": i + 1,
            "chunk_id": meta.get("chunk_id", f"chunk_{i}"),
            "source": meta.get("source", "—"),
            "page": meta.get("page", "—"),
            "section": meta.get("section_detected", "general"),
            "section_label": SECTION_LABELS.get(meta.get("section_detected", "general"), "general"),
            "char_count": meta.get("char_count", len(c.get("preview", ""))),
            "preview": c.get("preview", ""),
        })
    df = pd.DataFrame(rows)

    # ── Gráfico 1: distribución de secciones (donut) ────────────────────
    st.subheader("🗂️ Distribución por sección académica")
    sec_counts = df.groupby("section_label")["idx"].count().reset_index()
    sec_counts.columns = ["Sección", "Chunks"]
    sec_counts = sec_counts.sort_values("Chunks", ascending=False)

    col_pie, col_bar = st.columns([1, 1])
    with col_pie:
        fig_pie = px.pie(
            sec_counts,
            values="Chunks",
            names="Sección",
            hole=0.4,
            title="Proporción de chunks",
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar = px.bar(
            sec_counts,
            x="Chunks",
            y="Sección",
            orientation="h",
            color="Sección",
            title="Chunks por sección",
        )
        fig_bar.update_layout(showlegend=False, height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Gráfico 2: tamaño de chunks ──────────────────────────────────────
    st.subheader("📏 Distribución del tamaño de chunks")
    col_hist, col_scatter = st.columns([1, 1])

    with col_hist:
        fig_hist = px.histogram(
            df,
            x="char_count",
            nbins=20,
            title="Histograma de tamaños (caracteres)",
            labels={"char_count": "Caracteres por chunk"},
            color_discrete_sequence=["#2196F3"],
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_scatter:
        fig_scatter = px.scatter(
            df,
            x="idx",
            y="char_count",
            color="section_label",
            title="Tamaño por posición en el documento",
            labels={"idx": "Posición (chunk #)", "char_count": "Caracteres", "section_label": "Sección"},
            hover_data=["page", "chunk_id"],
        )
        fig_scatter.update_layout(height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Gráfico 3: chunks por página ────────────────────────────────────
    st.subheader("📖 Chunks generados por página")
    if "page" in df.columns:
        page_counts = df.groupby("page")["idx"].count().reset_index()
        page_counts.columns = ["Página", "Chunks"]
        page_counts = page_counts.sort_values("Página")
        fig_page = px.bar(
            page_counts,
            x="Página",
            y="Chunks",
            title="Número de chunks por página del PDF",
            color="Chunks",
            color_continuous_scale="Blues",
        )
        fig_page.update_layout(coloraxis_showscale=False, height=300)
        st.plotly_chart(fig_page, use_container_width=True)

    # ── Tabla interactiva de chunks ──────────────────────────────────────
    st.subheader("📋 Explorador de chunks")

    sections_available = ["Todas"] + sorted(df["section_label"].unique().tolist())
    filter_sec = st.selectbox("Filtrar por sección:", sections_available)
    search_text = st.text_input("🔎 Buscar en el texto del chunk:", "")

    df_filtered = df.copy()
    if filter_sec != "Todas":
        df_filtered = df_filtered[df_filtered["section_label"] == filter_sec]
    if search_text:
        df_filtered = df_filtered[df_filtered["preview"].str.contains(search_text, case=False, na=False)]

    st.caption(f"Mostrando {len(df_filtered)} de {len(df)} chunks cargados")

    for _, row in df_filtered.iterrows():
        color = SECTION_COLORS.get(row["section"], "#BDBDBD")
        with st.expander(
            f"#{row['idx']:03d} — Página {row['page']} — {row['char_count']} chars",
            expanded=False,
        ):
            st.markdown(
                f'**Sección:** {section_badge(row["section"])}  &nbsp;&nbsp;'
                f'**ID:** `{row["chunk_id"]}`  &nbsp;&nbsp;'
                f'**Fuente:** `{row["source"]}`',
                unsafe_allow_html=True,
            )
            st.markdown("---")
            st.markdown(f"```\n{row['preview']}\n```")


# ─────────────────────────────────────────────
#  RENDERIZADOR DE RESPUESTA FLOWISE
# ─────────────────────────────────────────────
def render_flowise_answer(answer_text: str):
    """Convierte la respuesta JSON de Flowise en tarjetas visuales legibles."""
    import json

    # Intentar parsear como JSON
    data = None
    if isinstance(answer_text, str):
        text = answer_text.strip()
        # Quitar posibles bloques markdown ```json ... ```
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            data = json.loads(text)
        except Exception:
            data = None

    if not isinstance(data, dict):
        # No es JSON estructurado → mostrarlo como markdown normal
        st.markdown(answer_text)
        return

    # ── Cabecera: puntuación + nivel ──────────────────────────────────────
    score  = data.get("puntuacion_general")
    nivel  = data.get("nivel_tesis", "").capitalize()

    NIVEL_COLOR = {
        "excelente":   "🟢",
        "muy bueno":   "🟢",
        "bueno":       "🔵",
        "aceptable":   "🟡",
        "regular":     "🟠",
        "deficiente":  "🔴",
        "insuficiente":"🔴",
    }
    emoji_nivel = NIVEL_COLOR.get(nivel.lower(), "⚪")

    col_score, col_nivel, col_spacer = st.columns([1, 1, 2])
    if score is not None:
        col_score.metric("📊 Puntuación general", f"{score} / 10")
    if nivel:
        col_nivel.markdown(
            f"**Nivel de tesis**\n\n"
            f"<span style='font-size:1.4em'>{emoji_nivel} {nivel}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Resumen ejecutivo ────────────────────────────────────────────────
    resumen = data.get("resumen_ejecutivo")
    if resumen:
        st.info(f"📋 **Resumen ejecutivo**\n\n{resumen}")

    # ── Mensaje pedagógico ───────────────────────────────────────────────
    mensaje = data.get("mensaje_pedagogico")
    if mensaje:
        st.success(f"💬 **Retroalimentación pedagógica**\n\n{mensaje}")

    # ── Puntos fuertes / Áreas de mejora ─────────────────────────────────
    col_f, col_m = st.columns(2)

    with col_f:
        st.markdown("### ✅ Puntos fuertes")
        puntos = data.get("puntos_fuertes", [])
        if puntos:
            for p in puntos:
                st.markdown(f"- {p}")
        else:
            st.caption("No se registraron puntos fuertes.")

    with col_m:
        st.markdown("### ⚠️ Áreas de mejora")
        areas = data.get("areas_mejora", [])
        if areas:
            for a in areas:
                st.markdown(f"- {a}")
        else:
            st.caption("No se detectaron áreas de mejora.")

    # ── Recomendaciones priorizadas ──────────────────────────────────────
    recomendaciones = data.get("recomendaciones_priorizadas", [])
    if recomendaciones:
        st.markdown("### 🎯 Recomendaciones priorizadas")
        for rec in sorted(recomendaciones, key=lambda r: r.get("prioridad", 99)):
            prioridad     = rec.get("prioridad", "—")
            recomendacion = rec.get("recomendacion", "")
            justificacion = rec.get("justificacion", "")
            with st.expander(f"**#{prioridad}** — {recomendacion}", expanded=prioridad == 1):
                if justificacion:
                    st.caption(f"💡 {justificacion}")

    # ── Siguiente paso ───────────────────────────────────────────────────
    siguiente = data.get("siguiente_paso")
    if siguiente:
        st.markdown("---")
        st.warning(f"🚀 **Siguiente paso recomendado**\n\n{siguiente}")


# ─────────────────────────────────────────────
#  PANTALLA 3 — CONSULTAR AGENTES
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  Helpers de extracción de datos de la respuesta
# ─────────────────────────────────────────────
# Mapeo nombre-de-agente → etiqueta normalizada usada en los 6 agentes Python
# y en los nodos del Agentflow Flowise (que comparten labels casi 1:1).
_AGENT_LABEL_TO_KEY = {
    "mentor intake":     "mentor_intake",
    "supervisor":        "mentor_intake",   # alias futuro (Sprint 4 re-etiquetado)
    "investigador":      "investigador",
    "auditor":           "auditor",
    "metodologico":      "metodologico",
    "metodológico":      "metodologico",
    "metodologo":        "metodologico",
    "metodólogo":        "metodologico",
    "redactor":          "redactor",
    "mentor final":      "mentor_final",
    "sintesis":          "mentor_final",
    "síntesis":          "mentor_final",
    "sintesis y consenso": "mentor_final",
}

# Mapeo de state-keys (formato 'returnCustomStateValues' del End node) →
# agent-keys del frontend. Habilitado por Commit 4 del Sprint 3: el End
# node devuelve directamente el flow state en lugar de sólo el último
# output, así que el frontend lee los 6 agentes con un único parseo en
# vez de caminar agentFlowExecutedData.
_STATE_KEY_TO_AGENT = {
    "intake_result":     "mentor_intake",
    "research_findings": "investigador",
    "audit_result":      "auditor",
    "method_result":     "metodologico",
    "writing_result":    "redactor",
    "mentor_result":     "mentor_final",
}


def _parse_maybe_json(text: Any) -> Any:
    """Intenta parsear text como JSON; si falla, lo devuelve tal cual."""
    if not isinstance(text, str):
        return text
    import json as _json

    # Quitar code fences ``` o ```json
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return _json.loads(cleaned)
    except Exception:
        return text


def _extract_agent_outputs(raw_result: dict) -> dict:
    """
    Devuelve un dict { 'mentor_intake': {...}, 'investigador': {...}, ... }
    con la salida parseada de cada uno de los 6 agentes, normalizando los
    dos modos:

      - Flowise: lee de raw_result['flowise_response']['agentFlowExecutedData']
                 cada nodo y matchea su nodeLabel contra _AGENT_LABEL_TO_KEY.
      - Python agentes: lee de raw_result['memory'][*] directamente.

    Para agentes ausentes, la key queda con dict vacío {}.
    """
    outputs: Dict[str, Any] = {k: {} for k in
        ["mentor_intake", "investigador", "auditor", "metodologico", "redactor", "mentor_final"]
    }

    # ── Modo Python agentes ──────────────────────────────────────────────
    memory = raw_result.get("memory", {})
    if isinstance(memory, dict) and memory:
        for k in outputs.keys():
            if k in memory:
                outputs[k] = memory[k] if isinstance(memory[k], dict) else _parse_maybe_json(memory[k])
        return outputs

    # ── Modo Flowise ─────────────────────────────────────────────────────
    flowise_resp = raw_result.get("flowise_response", {})
    if not isinstance(flowise_resp, dict):
        return outputs

    # ── Estrategia A — formato nuevo (Sprint 3 Commit 4) ────────────────
    # El End node 'returnCustomStateValues' devuelve un dict con las state
    # keys (intake_result, research_findings, etc.) directamente serializado
    # en flowise_response['text']. Cada valor es a su vez un JSON string.
    state_dict = _parse_maybe_json(flowise_resp.get("text") or "")
    if isinstance(state_dict, dict) and any(k in state_dict for k in _STATE_KEY_TO_AGENT):
        for state_key, agent_key in _STATE_KEY_TO_AGENT.items():
            raw_value = state_dict.get(state_key)
            if not raw_value:
                continue
            parsed = _parse_maybe_json(raw_value)
            outputs[agent_key] = (
                parsed if isinstance(parsed, dict)
                else {"raw_output": parsed}
            )
        return outputs

    # ── Estrategia B — formato legacy (returnLastOutput) ────────────────
    # Caminamos agentFlowExecutedData para extraer cada nodo individualmente.
    exec_data = flowise_resp.get("agentFlowExecutedData", []) or []
    for node in exec_data:
        if not isinstance(node, dict):
            continue
        label = (node.get("nodeLabel") or node.get("label") or "").strip().lower()
        key   = _AGENT_LABEL_TO_KEY.get(label)
        if not key:
            continue
        content = (
            node.get("data", {})
                .get("output", {})
                .get("content", "")
        )
        parsed = _parse_maybe_json(content)
        if parsed:
            outputs[key] = parsed if isinstance(parsed, dict) else {"raw_output": parsed}

    # Último recurso: el text del flowise_response es el output del Mentor Final.
    if not outputs["mentor_final"]:
        outputs["mentor_final"] = _parse_maybe_json(
            flowise_resp.get("text") or flowise_resp.get("output") or {}
        ) or {}

    return outputs


def _extract_score(final_data: dict) -> float:
    """Extrae puntuación general 0-10 del output del Mentor Final."""
    if not isinstance(final_data, dict):
        return 0.0
    val = (
        final_data.get("puntuacion_general")
        or final_data.get("puntuacion")
        or final_data.get("score")
        or 0
    )
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


# ─────────────────────────────────────────────
#  Render de cada pestaña
# ─────────────────────────────────────────────

def _render_tab_evaluation(agents: dict, final_data: dict, raw_result: dict) -> None:
    """Pestaña 1 — Evaluación: texto final, feedback auditor, recomendaciones."""
    texto_sugerido   = raw_result.get("texto_sugerido")
    original_context = raw_result.get("original_context", "")

    # ── Texto final (sugerido por Redactor / Mentor Final) ──────────────
    st.subheader("✏️ Texto final (versión mejorada)")
    if texto_sugerido:
        col_orig, col_sug = st.columns(2, gap="medium")
        with col_orig:
            st.markdown(
                "<p style='font-weight:600;color:#888'>📄 Texto original</p>",
                unsafe_allow_html=True,
            )
            st.text_area("original", value=original_context, height=320,
                         disabled=True, label_visibility="collapsed")
        with col_sug:
            st.markdown(
                "<p style='font-weight:600;color:#2e7d32'>✨ Texto sugerido</p>",
                unsafe_allow_html=True,
            )
            st.text_area("sugerido", value=texto_sugerido, height=320,
                         label_visibility="collapsed",
                         help="Selecciona todo (Ctrl+A) y copia.")
    else:
        st.info(
            "Texto sugerido no disponible. Verifica que `GROQ_API_KEY` esté "
            "configurada en `.streamlit/secrets.toml`."
        )

    # ── Feedback del Auditor ─────────────────────────────────────────────
    auditor = agents.get("auditor", {})
    if auditor:
        st.subheader("🔍 Feedback del Auditor")
        # El auditor puede devolver varios campos; mostramos comentario + flags + sugerencias
        comentario = auditor.get("comentario") or auditor.get("evaluacion") or ""
        if comentario:
            st.markdown(comentario)
        sugs = auditor.get("sugerencias", []) or auditor.get("recomendaciones", [])
        if sugs:
            st.markdown("**Sugerencias del Auditor:**")
            for s in sugs:
                st.markdown(f"- {s}")
        flags = auditor.get("flags", []) or auditor.get("issues", [])
        if flags:
            st.warning("⚠️ Flags detectados: " + ", ".join(map(str, flags)))

    # ── Expander: ¿Qué recomendaría el Redactor? ────────────────────────
    redactor = agents.get("redactor", {})
    if redactor:
        with st.expander("✍️ ¿Qué recomendaría el Redactor?"):
            comentario = redactor.get("comentario") or redactor.get("recomendacion") or ""
            if comentario:
                st.markdown(comentario)
            if not comentario:
                st.json(redactor)

    # ── Recomendaciones generales (Metodológico + Mentor Final) ─────────
    st.subheader("🎯 Recomendaciones generales")
    metodo = agents.get("metodologico", {})
    metodo_recs = metodo.get("recomendaciones", []) or metodo.get("sugerencias", [])
    final_recs  = final_data.get("recomendaciones_priorizadas", []) or \
                  final_data.get("recomendaciones", [])

    if metodo_recs:
        st.markdown("**Del Metodólogo:**")
        for r in metodo_recs:
            st.markdown(f"- {r}")

    if final_recs:
        st.markdown("**Priorizadas (del Mentor Final):**")
        for rec in sorted(final_recs, key=lambda r: r.get("prioridad", 99) if isinstance(r, dict) else 99):
            if isinstance(rec, dict):
                pri = rec.get("prioridad", "—")
                txt = rec.get("recomendacion", "")
                jus = rec.get("justificacion", "")
                with st.expander(f"**#{pri}** — {txt}", expanded=pri == 1):
                    if jus:
                        st.caption(f"💡 {jus}")
            else:
                st.markdown(f"- {rec}")

    if not (metodo_recs or final_recs):
        st.caption("Sin recomendaciones explícitas en esta evaluación.")


def _render_tab_debate(agents: dict, final_data: dict) -> None:
    """
    Pestaña 2 — Debate (mapeo provisional con los 6 agentes actuales).

    Hasta Sprint 4 los nodos no producen Consenso/Disenso explícitos, por
    lo que mapeamos:
      - Perspectiva Formal       = Auditor
      - Perspectiva Metodológica = Metodológico
      - Perspectiva Contextual   = Investigador
      - Síntesis                 = Mentor Final
    """
    st.caption(
        "_Mapeo provisional desde los 6 agentes actuales. Consenso/Disenso "
        "explícitos llegan al rediseñar agentes (Sprint 4)._"
    )

    perspectives = [
        ("📐 Perspectiva Formal",       agents.get("auditor",       {}), "rigor académico, normas, citas"),
        ("🧪 Perspectiva Metodológica", agents.get("metodologico",  {}), "diseño, instrumentos, validez"),
        ("🔬 Perspectiva Contextual",   agents.get("investigador",  {}), "antecedentes, estado del arte"),
        ("🧭 Síntesis",                  agents.get("mentor_final",  {}) or final_data, "integración del debate"),
    ]

    for label, data, hint in perspectives:
        with st.expander(label, expanded=False):
            st.caption(f"_{hint}_")
            if not data:
                st.caption("— Sin output disponible.")
                continue
            comentario = (
                data.get("comentario")
                or data.get("evaluacion")
                or data.get("evaluacion_inicial")
                or data.get("mensaje_pedagogico")
                or data.get("resumen_ejecutivo")
                or ""
            )
            if comentario:
                st.markdown(comentario)
            else:
                st.json(data)

    st.markdown("---")
    st.markdown("**🟢 Consenso**")
    st.caption("_Disponible al rediseñar agentes (Sprint 4 — re-etiquetado del Mentor Final como 'Síntesis y Consenso')._")
    st.markdown("**🔴 Disenso**")
    st.caption("_Disponible al rediseñar agentes (Sprint 4)._")


def _render_tab_rag(result: dict) -> None:
    """Pestaña 3 — Contexto RAG con sub-pestañas (libros y cruzado en Sprint 4)."""
    sub_tesis, sub_libros, sub_cruzado = st.tabs([
        "📄 Del PDF de tesis",
        "📚 De libros de referencia",
        "🔗 Contexto cruzado",
    ])

    with sub_tesis:
        context = result.get("context_preview", "")
        if context:
            st.caption(f"Fragmentos recuperados: **{result.get('chunks_retrieved', '—')}**")
            st.text_area("contexto_tesis", value=context, height=320, disabled=True,
                         label_visibility="collapsed")
        else:
            st.info("No hay contexto recuperado disponible en este resultado.")

    with sub_libros:
        st.caption(
            "_Disponible en Sprint 4: indexamos los libros metodológicos "
            "(Hernández Sampieri y otros) y mostramos los fragmentos relevantes._"
        )

    with sub_cruzado:
        st.caption(
            "_Disponible en Sprint 4: pasajes del PDF que se contrastan con "
            "principios de los libros de referencia._"
        )


def _render_tab_reportes(
    question: str,
    result: dict,
    agents: dict,
    final_data: dict,
) -> None:
    """Pestaña 4 — Reportes (métricas NLP + 3 descargas)."""
    import json as _json

    # ── Métricas NLP ─────────────────────────────────────────────────────
    st.subheader("📊 Métricas NLP")
    st.caption(
        "Comparan el _texto original_ analizado vs el _texto sugerido_ "
        "(reescritura propuesta por el pipeline)."
    )

    texto_sugerido   = result.get("result", {}).get("texto_sugerido")
    original_context = result.get("result", {}).get("original_context", "") \
        or result.get("context_preview", "")

    metrics = st.session_state.get("last_metrics")

    if not texto_sugerido or not original_context:
        st.info("Las métricas requieren texto original + sugerido. No hay datos suficientes.")
    elif metrics is None:
        if st.button("🧮 Calcular métricas NLP", type="primary"):
            with st.spinner("Calculando ROUGE / BLEU / similitud coseno…"):
                from services.metrics_service import compute_all
                score = _extract_score(final_data)
                # Sin "score_before" real, usamos 5.0 como baseline neutral.
                # Cuando Sprint 4 entregue iteración previa, lo cambiamos.
                metrics = compute_all(
                    reference=original_context,
                    hypothesis=texto_sugerido,
                    score_before=5.0,
                    score_after=score,
                )
                st.session_state["last_metrics"] = metrics
                st.rerun()
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROUGE-1",      _fmt(metrics.get("rouge1")))
        m2.metric("ROUGE-2",      _fmt(metrics.get("rouge2")))
        m3.metric("ROUGE-L",      _fmt(metrics.get("rougeL")))
        m4.metric("BLEU",         _fmt(metrics.get("bleu")))
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Cos sim",      _fmt(metrics.get("cosine_similarity")))
        m6.metric("Gain Score",   _fmt(metrics.get("gain_score")))
        m7.metric("Kappa",        _fmt(metrics.get("kappa")))
        m8.metric("Puntaje 0-10", f"{_extract_score(final_data):.1f}")
        if metrics.get("kappa") is None:
            st.caption("_Kappa: requiere 2 evaluaciones — habilitable cuando Sprint 4 active iteraciones múltiples._")

    st.markdown("---")

    # ── Descargas ────────────────────────────────────────────────────────
    st.subheader("⬇️ Descargas")

    ciclo = {
        "question":          question,
        "mode":              result.get("mode"),
        "chunks_retrieved":  result.get("chunks_retrieved"),
        "elapsed_seconds":   result.get("elapsed_seconds"),
        "agents":            agents,
        "final_evaluation":  final_data,
        "texto_sugerido":    texto_sugerido,
        "original_context":  original_context,
        "thread_id":         st.session_state.get("thread_id"),
        "rubric_id":         st.session_state.get("rubric_id"),
        "iterations":        st.session_state.get("iterations"),
    }
    debate_md  = _build_debate_markdown(agents, final_data)
    metricas   = metrics or {"info": "No calculadas todavía. Pulsa 'Calcular métricas NLP'."}

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📄 ciclo.json",
            data=_json.dumps(ciclo, ensure_ascii=False, indent=2),
            file_name="ciclo.json",
            mime="application/json",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "💬 debate.md",
            data=debate_md,
            file_name="debate.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col3:
        st.download_button(
            "📊 metricas.json",
            data=_json.dumps(metricas, ensure_ascii=False, indent=2),
            file_name="metricas.json",
            mime="application/json",
            use_container_width=True,
        )


def _fmt(v: Any) -> str:
    """Formatea valores numéricos para st.metric; None → '—'."""
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _build_debate_markdown(agents: dict, final_data: dict) -> str:
    """Genera el markdown descargable de la Pestaña 2 (Debate)."""
    import json as _json
    lines = ["# Debate Multiagente", ""]
    perspectives = [
        ("Perspectiva Formal (Auditor)",       agents.get("auditor",      {})),
        ("Perspectiva Metodológica (Metodólogo)", agents.get("metodologico", {})),
        ("Perspectiva Contextual (Investigador)", agents.get("investigador", {})),
        ("Síntesis (Mentor Final)",            agents.get("mentor_final", {}) or final_data),
    ]
    for label, data in perspectives:
        lines.append(f"## {label}")
        if not data:
            lines.append("_(sin output disponible)_")
        else:
            comentario = (
                data.get("comentario")
                or data.get("evaluacion")
                or data.get("evaluacion_inicial")
                or data.get("mensaje_pedagogico")
                or ""
            )
            if comentario:
                lines.append(comentario)
            else:
                lines.append("```json\n" + _json.dumps(data, ensure_ascii=False, indent=2) + "\n```")
        lines.append("")
    lines.append("---")
    lines.append("_Consenso/Disenso: disponibles al rediseñar agentes (Sprint 4)._")
    return "\n".join(lines)


def _render_query_result_block(
    question: str,
    result: dict,
    elapsed: float,
) -> None:
    """
    Renderiza el resultado de una evaluación completada en 4 pestañas:
      1. Evaluación  — texto final, feedback auditor, recomendaciones.
      2. Debate      — perspectivas mapeadas a los 6 agentes actuales.
      3. Contexto RAG— sub-pestañas: tesis / libros / cruzado.
      4. Reportes    — métricas NLP + descargas (ciclo/debate/metricas).

    Extraído de page_query para poder volver a mostrarlo entre reruns
    leyendo desde st.session_state['last_result'].
    """
    raw_result    = result.get("result", {})
    agent_outputs = _extract_agent_outputs(raw_result)
    final_data    = agent_outputs.get("mentor_final", {}) or {}

    # ── Header: "Mentoría Completada" + 3 métricas ─────────────────────
    st.success("✅ **Mentoría Completada**")

    iterations = st.session_state.get("iterations", 1)
    score      = _extract_score(final_data)
    vigesimal  = round(score * 2, 1)

    h1, h2, h3 = st.columns(3)
    h1.metric("Iteraciones",     iterations)
    h2.metric("Puntaje (0-10)",  f"{score:.1f}")
    h3.metric("Nota vigesimal",  f"{vigesimal:.1f}")
    st.caption(
        f"Modo: `{result.get('mode', '—')}` · Chunks: {result.get('chunks_retrieved', '—')} · "
        f"Tiempo: {result.get('elapsed_seconds', elapsed)} s"
    )

    st.markdown("---")

    # ── 4 pestañas ──────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Evaluación", "💬 Debate", "📖 Contexto RAG", "📊 Reportes",
    ])

    with tab1:
        _render_tab_evaluation(agent_outputs, final_data, raw_result)
    with tab2:
        _render_tab_debate(agent_outputs, final_data)
    with tab3:
        _render_tab_rag(result)
    with tab4:
        _render_tab_reportes(question, result, agent_outputs, final_data)

    # ── Debug expandible (payload completo) ─────────────────────────────
    with st.expander("🔧 Ver payload completo (debug)", expanded=False):
        st.json(result)


_OVERVIEW_SECTION_ID = "__overview__"


def _build_question_from_section(section_id: str, section_title: str = "") -> str:
    """
    Construye la pregunta enviada al backend a partir de la sección elegida
    en el dropdown. Reemplaza el text_area libre de versiones previas.
    """
    if section_id == _OVERVIEW_SECTION_ID:
        return (
            "Evalúa de forma integral el proyecto de tesis: planteamiento del problema, "
            "marco teórico, metodología, coherencia entre objetivos y resultados, y "
            "rigor académico general. Aplica la rúbrica UPAO de Ing. Sistemas."
        )
    label = f"{section_id} {section_title}".strip()
    return (
        f"Evalúa la sección '{label}' del proyecto de tesis aplicando la rúbrica "
        f"UPAO de Ing. Sistemas. Identifica fortalezas, debilidades y "
        f"recomendaciones específicas."
    )


def _render_query_form_block(total_chunks: int) -> tuple[str, int, str | None, bool]:
    """
    Renderiza el Paso 2 — Configura y lanza la evaluación.
    Devuelve (question, top_k, session_id, send_clicked).

    Layout (alineado con la app de referencia):
      - Banner verde 'PDF cargado: <nombre>'
      - Linea 'Rubrica activa: UPAO oficial (N items).'
      - H2 'Paso 2 — Configura y lanza la evaluacion'
      - Dropdown de seccion (overview + outline)
      - Texto contextual
      - Expander 'Configuracion avanzada' con slider iteraciones (1-3, default 2)
        y top_k (oculto en avanzado, no expuesto en la UX principal)
      - Banner informativo con tiempo estimado y agentes
      - Boton rojo 'Iniciar Evaluacion Multiagente'
    """
    pdf_name      = st.session_state.get("pdf_filename", "—")
    rubric        = RUBRICS.get(st.session_state.get("rubric_id", "upao_ing_sistemas"), {})
    rubric_label  = rubric.get("label",   "—")
    rubric_items  = rubric.get("items",   0)
    custom_rubric = st.session_state.get("custom_rubric_filename", "")
    outline       = st.session_state.get("pdf_outline", []) or []

    st.success(f"📄 **PDF cargado:** `{pdf_name}`")
    if custom_rubric:
        st.markdown(f"_Rúbrica activa: **{custom_rubric}** (personalizada)._")
    else:
        st.markdown(f"_Rúbrica activa: **{rubric_label}** ({rubric_items} ítems)._")

    st.header("Paso 3 — Selecciona la sección a evaluar")

    # ── Dropdown de sección ──────────────────────────────────────────────
    # Construimos opciones: primero "Vista general", después outline.
    option_ids:   list[str] = [_OVERVIEW_SECTION_ID]
    option_labels: list[str] = ["Vista general del proyecto (panorama completo)"]
    for h in outline:
        option_ids.append(h["section_id"])
        option_labels.append(f"{h['section_id']} — {h['title']}")

    current_sid = st.session_state.get("selected_section_id", _OVERVIEW_SECTION_ID)
    try:
        current_idx = option_ids.index(current_sid)
    except ValueError:
        current_idx = 0

    chosen_label = st.selectbox(
        "Sección del proyecto de tesis",
        options=option_labels,
        index=current_idx,
        help=(
            "Elige una sección específica para análisis profundo, o 'Vista general' "
            "para una evaluación integral del proyecto."
        ),
    )
    selected_idx = option_labels.index(chosen_label)
    selected_sid = option_ids[selected_idx]
    selected_title = (
        "" if selected_sid == _OVERVIEW_SECTION_ID
        else outline[selected_idx - 1]["title"]   # -1 por la entrada overview
    )
    st.session_state["selected_section_id"] = selected_sid

    # Texto contextual debajo del dropdown
    if selected_sid == _OVERVIEW_SECTION_ID:
        st.caption(
            "🔍 El sistema recuperará fragmentos representativos de todas las secciones "
            "y los agentes producirán una evaluación integral."
        )
    else:
        st.caption(
            f"🔍 Análisis enfocado en la sección **{selected_sid} {selected_title}**. "
            "Los agentes profundizarán en fortalezas, debilidades y mejoras concretas."
        )

    # ── Configuración avanzada ────────────────────────────────────────────
    with st.expander("⚙️ Configuración avanzada"):
        iterations = st.slider(
            "Iteraciones del panel de debate",
            min_value=1, max_value=3,
            value=st.session_state.get("iterations", 2),
            help=(
                "Número de pasadas del panel multiagente. Más iteraciones = mayor "
                "profundidad de análisis, pero también mayor latencia y costo de tokens."
            ),
        )
        st.session_state["iterations"] = iterations

        top_k = st.slider(
            "Top-K (fragmentos RAG)",
            min_value=1, max_value=20, value=5,
            help="Cuántos fragmentos relevantes de ChromaDB se le pasan al agente como contexto.",
        )

        # Por defecto usamos el thread_id del workflow para que Flowise mantenga
        # contexto entre consultas del mismo PDF.
        session_id = st.text_input(
            "Session ID (avanzado)",
            value=st.session_state.get("thread_id", ""),
            help="Identifica la conversación en Flowise. Por defecto, el thread_id del workflow.",
        )

    # ── Banner informativo ───────────────────────────────────────────────
    iters    = st.session_state.get("iterations", 2)
    eta_min  = 1 if iters == 1 else (2 if iters == 2 else 3)
    eta_max  = 2 if iters == 1 else (3 if iters == 2 else 5)
    st.info(
        f"🛈 **{iters} iteración(es)** · panel de debate (4 subagentes) · "
        f"Tiempo estimado: **{eta_min}–{eta_max} min** · "
        f"Agentes: _Supervisor, Auditor, Metodólogo, Consenso, Disenso, Debate, Redactor_"
    )

    # ── Botón rojo grande ────────────────────────────────────────────────
    # Streamlit no soporta color rojo nativo en st.button; usamos type='primary'
    # y un divider visual para diferenciarlo del resto del form.
    st.markdown("")
    send = st.button(
        "🚀 Iniciar Evaluación Multiagente",
        type="primary",
        use_container_width=True,
    )

    # Construir la pregunta a enviar al backend
    question = _build_question_from_section(selected_sid, selected_title)

    return question, top_k, session_id or None, send


def page_query():
    # ── Verificar que hay datos ──────────────────────────────────────────
    col_info = api_collection_info()
    if col_info is None:
        st.error("No se puede conectar con el backend.")
        return

    total = col_info.get("total_chunks", 0)
    if total == 0:
        st.warning("⚠️ No hay ningún proyecto de investigación cargado. Primero sube un PDF.")
        if st.button("← Volver a cargar PDF"):
            st.session_state["workflow_stage"] = STAGE_UPLOAD
            st.rerun()
        return

    # ── Si hay un resultado guardado → mostrarlo (sobrevive reruns) ─────
    if (
        st.session_state.get("workflow_stage") == STAGE_RESULTS
        and st.session_state.get("last_result") is not None
    ):
        stored = st.session_state["last_result"]
        _render_query_result_block(
            question=st.session_state.get("last_question", ""),
            result=stored,
            elapsed=stored.get("elapsed_seconds", 0.0),
        )
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔁 Hacer otra consulta (mismo PDF)", use_container_width=True):
                reset_for_new_section()
                st.rerun()
        with col_b:
            if st.button("🆕 Nueva evaluación (otro PDF)", use_container_width=True):
                # Misma logica que el boton del sidebar: vaciar ChromaDB para
                # poder volver a cargar PDF sin que el bridge nos atrape.
                api_reset_collection()
                reset_all_state()
                st.rerun()
        return

    # ── Formulario Paso 2 (dropdown sección + slider + botón) ──────────
    question, top_k, session_id, send = _render_query_form_block(total)

    # ── Ejecutar consulta + persistir resultado + avanzar al stage RESULTS
    if send and len(question.strip()) >= 5:
        with st.spinner(
            "Los agentes están analizando el proyecto de investigación… "
            "puede tardar entre 1 y 5 minutos (más si Groq aplica rate-limiting)."
        ):
            t0 = time.time()
            result, status = api_query(question.strip(), top_k=top_k, session_id=session_id)
            elapsed = round(time.time() - t0, 1)

        if status == 200:
            # Persistir para que sobreviva reruns y avanzar workflow → RESULTS.
            # En la próxima ejecución de page_query, el branch del top renderiza
            # _render_query_result_block leyendo desde session_state.
            st.session_state["last_question"] = question.strip()
            st.session_state["last_result"]   = {**result, "elapsed_seconds": elapsed}
            st.session_state["workflow_stage"] = STAGE_RESULTS

            # Guardar también en el historial (preexistente)
            st.session_state.setdefault("query_history", []).append(
                {
                    "question": question.strip(),
                    "elapsed": elapsed,
                    "chunks_retrieved": result.get("chunks_retrieved"),
                    "mode": result.get("mode"),
                }
            )
            st.rerun()

        else:
            st.error(f"❌ Error ({status}): {result.get('detail', 'Error desconocido')}")
            if status == 404:
                st.info("Asegúrate de haber subido un PDF.")
            elif status == 502:
                st.info(
                    "Flowise no está respondiendo. Verifica que esté corriendo y "
                    "que FLOWISE_CHATFLOW_ID sea correcto."
                )
            elif status == 504:
                st.info(
                    "💡 **Sugerencias para acelerar la consulta:**\n"
                    "- Reduce **Top-K** en parámetros avanzados (menos contexto = menos tokens).\n"
                    "- Si tu plan de Groq es Free, considera actualizar al Dev Tier en "
                    "[console.groq.com/settings/billing](https://console.groq.com/settings/billing) "
                    "para subir el límite TPM de 6 000 a 30 000+.\n"
                    "- También puedes desactivar Flowise con `USE_FLOWISE=false` en `.env` "
                    "para saltar el primer intento (90 s) y usar directamente los agentes Python."
                )

    # ── Historial de consultas ───────────────────────────────────────────
    if st.session_state.get("query_history"):
        st.markdown("---")
        st.subheader("📜 Historial de consultas (esta sesión)")
        hist_df = pd.DataFrame(st.session_state["query_history"])
        hist_df.index = hist_df.index + 1
        st.dataframe(hist_df, use_container_width=True)


# ─────────────────────────────────────────────
#  LAYOUT PRINCIPAL
# ─────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Evaluador de Proyecto de Investigación — RAG Multiagente",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inicializa st.session_state (thread_id, workflow_stage, rubric, etc.)
    init_session_state()

    # ── Sidebar persistente (reemplaza el st.radio anterior) ─────────────
    backend_ok = render_sidebar()

    # ── Contenido principal ───────────────────────────────────────────────
    if not backend_ok:
        st.error(
            "### ⛔ Backend no disponible\n\n"
            "El backend FastAPI corre embebido dentro de esta misma app Streamlit, "
            "pero falló al inicializar.\n\n"
            "**Causas comunes:**\n"
            "- Faltan **Secrets** en Streamlit Cloud (Settings → Secrets). "
            "Copia el contenido de `.streamlit/secrets.toml` allí.\n"
            "- Primera carga: el modelo `multilingual-e5-small` (~470 MB) se está descargando — "
            "espera ~30 s y recarga.\n"
            "- Memoria insuficiente: el modelo necesita ~500 MB libres."
        )
        return

    # ── Recuperación de sesión: si el usuario refresca la página y ya hay
    # chunks en ChromaDB, saltamos directo a 'configure' en lugar de
    # devolverlo al uploader (que rechazaría el mismo archivo).
    if st.session_state["workflow_stage"] == STAGE_UPLOAD:
        col_info = api_collection_info()
        if col_info and col_info.get("total_chunks", 0) > 0:
            st.session_state["workflow_stage"] = STAGE_CONFIGURE

    # ── Dispatcher por workflow_stage ────────────────────────────────────
    stage = st.session_state["workflow_stage"]
    if stage == STAGE_UPLOAD:
        page_upload()
    elif stage == STAGE_EMBEDDINGS:
        page_embeddings()
    else:  # STAGE_CONFIGURE | STAGE_RESULTS — ambos los renderiza page_query
        page_query()


if __name__ == "__main__":
    main()
