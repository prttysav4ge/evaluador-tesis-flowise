"""
Servicio de procesamiento de PDFs.

Pipeline TOC-aware (camino principal, portado de langgraph):
  1. Extrae texto con pdfplumber separando el ÍNDICE del contenido real
     (las páginas de índice se excluyen del RAG — no son contenido evaluable).
  2. Parsea el índice (TOC) → {nombre_sección: página_inicio}.
  3. Agrupa TODO el texto de cada sección (a través de varias páginas) ANTES
     de trocear, para que los chunks respeten los límites de sección.
  4. Trocea dentro de cada sección con RecursiveCharacterTextSplitter.
  5. Retorna chunks con metadatos (`seccion`, `pagina_inicio`, …) listos para
     guardar en ChromaDB.

Fallback (PDF sin índice parseable): comportamiento clásico página-a-página con
detección de sección por keyword + outline jerárquico 1.1.1.
"""
from __future__ import annotations

import io
import re
import logging
import warnings
from typing import Any, Dict, List, Tuple

import pdfplumber
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
#  Patrones de secciones académicas en español e inglés                  #
# ---------------------------------------------------------------------- #
_SECTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(RESUMEN|ABSTRACT)\b", re.I), "resumen"),
    (re.compile(r"\b(INTRODUCCION|INTRODUCTION|INTRODUCCIÓN)\b", re.I), "introduccion"),
    (re.compile(r"\bPLANTEAMIENTO\s+(DEL\s+)?PROBLEMA\b", re.I), "planteamiento_problema"),
    (re.compile(r"\bJUSTIFICACI[OÓ]N\b", re.I), "justificacion"),
    (re.compile(r"\bOBJETIVOS?\b", re.I), "objetivos"),
    (re.compile(r"\bHIP[OÓ]TESIS\b", re.I), "hipotesis"),
    (re.compile(r"\bANTECEDENTES\b", re.I), "antecedentes"),
    (re.compile(r"\bESTADO DEL ARTE\b", re.I), "estado_del_arte"),
    (re.compile(r"\bMARCO\s+TE[OÓ]RICO\b", re.I), "marco_teorico"),
    (re.compile(r"\bMARCO\s+CONCEPTUAL\b", re.I), "marco_conceptual"),
    (re.compile(r"\bMARCO\s+METODOL[OÓ]GICO\b", re.I), "marco_metodologico"),
    (re.compile(r"\bMETODOLOG[IÍ]A\b", re.I), "metodologia"),
    (re.compile(r"\bDISE[NÑ]O\s+(DE\s+)?INVESTIGACI[OÓ]N\b", re.I), "diseno_investigacion"),
    (re.compile(r"\bRESULTADOS?\b", re.I), "resultados"),
    (re.compile(r"\bAN[AÁ]LISIS\b", re.I), "analisis"),
    (re.compile(r"\bDISCUSI[OÓ]N\b", re.I), "discusion"),
    (re.compile(r"\bCONCLUSIONES?\b", re.I), "conclusiones"),
    (re.compile(r"\bBIBLIOGRAF[IÍ]A|REFERENCIAS\b", re.I), "referencias"),
]

# ---------------------------------------------------------------------- #
#  Detección jerárquica por numeración 1.1.1                              #
# ---------------------------------------------------------------------- #
# Encabezados de sección con numeración jerárquica: 1.1, 1.1.1, 1.1.1.1.
# Exigimos AL MENOS un punto (i.e. 2+ niveles) porque "9 SINOPSIS" o
# "13 Capítulo" suelen ser numeros de pagina pegados al body por pypdf,
# no encabezados reales. Las secciones de nivel-1 sin numeración tipo
# 'INTRODUCCIÓN', 'METODOLOGÍA' las captura igualmente _SECTION_PATTERNS.
# El título debe empezar con mayúscula o letra y tener 3-100 chars.
# Se aplica con re.MULTILINE sobre el texto limpio de cada página.
_HIERARCHICAL_HEADING_RE = re.compile(
    r"^[ \t]*(\d{1,2}(?:\.\d{1,2}){1,3})\.?\s+([A-ZÁÉÍÓÚÑa-zá-úñ][^\n]{2,99})[ \t]*$",
    re.MULTILINE,
)


def _looks_like_bibliography_entry(title: str) -> bool:
    """
    Filtro heurístico: descarta líneas que parecen citas bibliográficas
    (autores con año entre paréntesis) en lugar de títulos de sección.
    """
    return bool(re.search(r"\(\s*(?:19|20)\d{2}", title))


# Líneas del índice (Table of Contents) tienen formato "Título ............ 15".
# Detectamos 4+ puntos consecutivos como marcador robusto.
_TOC_DOT_LEADER_RE = re.compile(r"\.{4,}")


def _looks_like_toc_entry(title: str) -> bool:
    """True si la línea parece una entrada del índice (tiene dot leaders)."""
    return bool(_TOC_DOT_LEADER_RE.search(title))


# Items del cronograma/calendario que el regex puede confundir con secciones
# (ej. "7.1 Fecha de inicio", "7.2 Fecha de término" dentro del anexo de
# cronograma). No son secciones evaluables — son metadata del proyecto.
_NON_EVALUABLE_TITLE_RE = re.compile(
    r"^(?:"
    r"Fecha\s+(?:de\s+)?(?:inicio|t[eé]rmino|fin(?:alizaci[oó]n)?|entrega)"
    r"|Per[ií]odo\s+(?:de\s+)?(?:ejecuci[oó]n|estudio|investigaci[oó]n)"
    r"|Plazo\s+(?:de\s+)?(?:entrega|ejecuci[oó]n)"
    r"|Duraci[oó]n\s+(?:del\s+)?(?:proyecto|estudio)"
    r")\b",
    re.IGNORECASE,
)


def _looks_like_non_evaluable_metadata(title: str) -> bool:
    """
    True si el título es metadata del proyecto (fechas, plazos, duración)
    en lugar de una sección evaluable por el panel multiagente.
    """
    return bool(_NON_EVALUABLE_TITLE_RE.match(title.strip()))


def _clean_heading_title(title: str) -> str:
    """
    Limpia el título: quita dot leaders residuales y número de página al
    final ('Título .... 15' → 'Título'). Idempotente.
    """
    # Recortar todo desde el primer bloque de 2+ puntos consecutivos
    title = re.split(r"\s*\.{2,}.*$", title, maxsplit=1)[0]
    # Quitar número de página suelto al final ('Título 15')
    title = re.sub(r"\s+\d{1,4}\s*$", "", title)
    return title.strip()


def extract_hierarchical_outline(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detecta encabezados jerárquicos (1.1.1) en el texto de cada página y
    construye un outline ordenado por aparición.

    Returns:
        [{"section_id": "1.1.1", "title": "Antecedentes", "page": 12, "level": 3}, ...]

    Reglas para evitar falsos positivos:
      - Líneas con dot leaders (4+ puntos consecutivos) se asumen TOC y se descartan.
      - Citas bibliográficas con año entre paréntesis se descartan.
      - Si un mismo section_id aparece varias veces, se conserva la ÚLTIMA
        ocurrencia (la del cuerpo del documento, no la del índice).

    Si el PDF no usa numeración, retorna lista vacía → el caller cae a keywords.
    """
    # Indexamos por section_id para conservar la última aparición.
    by_id: Dict[str, Dict[str, Any]] = {}

    for page_data in pages:
        page_num = page_data["page"]
        text     = page_data["text"]

        for match in _HIERARCHICAL_HEADING_RE.finditer(text):
            section_id = match.group(1).rstrip(".")
            raw_title  = match.group(2).strip()

            if _looks_like_bibliography_entry(raw_title):
                continue
            if _looks_like_toc_entry(raw_title):
                continue   # es una línea del índice

            title = _clean_heading_title(raw_title)
            if len(title) < 3:
                continue   # quedó vacío tras la limpieza
            if _looks_like_non_evaluable_metadata(title):
                continue   # fechas/plazos del cronograma, no secciones

            by_id[section_id] = {
                "section_id": section_id,
                "title":      title,
                "page":       page_num,
                "level":      section_id.count(".") + 1,
            }

    # Ordenar por (página, id_natural) — los headings van en el orden del PDF.
    def _natural_sort_key(h: Dict[str, Any]) -> tuple:
        parts = [int(p) for p in h["section_id"].split(".")]
        return (h["page"], parts)

    outline = sorted(by_id.values(), key=_natural_sort_key)
    logger.info(f"📑 Outline jerárquico detectado: {len(outline)} encabezados")
    return outline


def _assign_chunks_to_outline(
    chunks: List[Dict[str, Any]],
    outline: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Asigna cada chunk al heading más reciente cuyo `page` sea ≤ al `page`
    del chunk. Esto agrupa los chunks bajo su sección padre.

    Modifica `chunks` in-place añadiendo `metadata["outline_section_id"]`,
    y devuelve un outline enriquecido con `chunks_count` + `chars_count`.

    Limitación: cuando una página tiene más de un heading, todos los chunks
    de esa página se asignan al ÚLTIMO heading anterior o de la propia página.
    Se acepta esta imprecisión a cambio de no requerir offsets dentro de la página.
    """
    if not outline:
        return []

    sorted_outline = sorted(outline, key=lambda h: h["page"])
    stats = {h["section_id"]: {"chunks": 0, "chars": 0} for h in sorted_outline}

    for chunk in chunks:
        page = chunk["metadata"].get("page", 0)
        current = None
        for heading in sorted_outline:
            if heading["page"] <= page:
                current = heading
            else:
                break
        if current is None:
            continue   # chunk anterior al primer heading; queda sin asignar
        sid = current["section_id"]
        chunk["metadata"]["outline_section_id"] = sid
        stats[sid]["chunks"] += 1
        stats[sid]["chars"]  += chunk["metadata"].get("char_count", 0)

    return [
        {**h, "chunks_count": stats[h["section_id"]]["chunks"],
              "chars_count":  stats[h["section_id"]]["chars"]}
        for h in sorted_outline
    ]


# ====================================================================== #
#  EXTRACCIÓN TOC-AWARE (portado de langgraph backend/rag/extractor.py)   #
# ====================================================================== #
# Línea de índice CON puntos de relleno: "Texto cualquiera .............. 12"
_RE_LINEA_TOC_DOTS = re.compile(r'\.{4,}\s*\d{1,4}\s*$')
# Entrada de índice CON o SIN puntos de relleno. Muchos PDF exportados desde
# Word alinean el número de página con tabulación/espacios en vez de dot leaders:
#     "1.1. Descripción y delimitación del problema      6"   (sin puntos)
#     "2.2 Base teórica .......................... 17"        (con puntos)
# Estructura: numbering + título (empieza en letra) + (dots | espacios) + nº pág.
_RE_ENTRADA_TOC = re.compile(
    r'^\s*(\d[\d.]*\.?\s*[A-Za-zÁÉÍÓÚÜÑáéíóúüñ][^\n]*?)(?:\.{2,}\s*|\s+)(\d{1,4})\s*$'
)
# Si más del 28 % de las líneas de una página tienen patrón TOC, se considera índice
_UMBRAL_PAGINA_TOC = 0.28


def _es_linea_toc(linea: str) -> bool:
    """True si la línea parece una entrada de índice (con o sin dot leaders)."""
    return bool(_RE_LINEA_TOC_DOTS.search(linea) or _RE_ENTRADA_TOC.match(linea))


def _ratio_lineas_toc(texto_pagina: str) -> float:
    """Retorna la fracción de líneas no vacías que tienen patrón de índice."""
    lineas = [l.strip() for l in texto_pagina.split('\n') if l.strip()]
    if len(lineas) < 2:
        return 0.0
    n_toc = sum(1 for l in lineas if _es_linea_toc(l))
    return n_toc / len(lineas)


def _parsear_toc(paginas_toc: List[str]) -> Dict[str, int]:
    """
    Extrae la estructura del índice: {nombre_seccion: numero_pagina}.
    Solo captura entradas que empiezan con número (secciones numeradas).
    """
    estructura: Dict[str, int] = {}
    for texto in paginas_toc:
        for linea in texto.split('\n'):
            m = _RE_ENTRADA_TOC.match(linea.strip())
            if m:
                nombre = re.sub(r'\s+', ' ', m.group(1)).strip()
                # Quitar dot leaders / espacios sobrantes al final del título
                nombre = re.sub(r'[\s.]{2,}$', '', nombre).strip()
                try:
                    pagina = int(m.group(2))
                    if nombre:
                        estructura[nombre] = pagina
                except ValueError:
                    pass
    return estructura


def extraer_contenido_sin_indice(
    pdf_bytes: bytes,
) -> Tuple[List[Tuple[int, str]], Dict[str, int]]:
    """
    Extrae el texto del PDF separando contenido real de páginas de índice/TOC.

    Estrategia:
      1. Analiza cada página por separado (pdfplumber).
      2. Las páginas donde ≥28 % de líneas tienen patrón "texto......N"
         se clasifican como TOC y se usan para parsear la estructura.
      3. Las páginas de contenido se retornan como lista (numero_pagina, texto).

    Returns:
        paginas_contenido: lista de (numero_pagina_1indexed, texto_pagina).
        estructura_toc:    dict {nombre_sección → numero_pagina_inicio}.
    """
    paginas_contenido: List[Tuple[int, str]] = []
    paginas_toc_texto: List[str] = []
    n_toc = 0

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Cannot set.*color")
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            total = len(pdf.pages)
            for i, pagina in enumerate(pdf.pages):
                numero_pagina = i + 1  # 1-indexed, coincide con números del TOC
                texto = pagina.extract_text()
                if not texto or not texto.strip():
                    continue

                ratio = _ratio_lineas_toc(texto)
                if ratio >= _UMBRAL_PAGINA_TOC:
                    n_toc += 1
                    paginas_toc_texto.append(texto)
                    logger.info(
                        f"Pág. {numero_pagina}/{total}: ÍNDICE (ratio={ratio:.2f}) — excluida del RAG"
                    )
                else:
                    paginas_contenido.append((numero_pagina, texto.strip()))

    estructura_toc = _parsear_toc(paginas_toc_texto)

    logger.info(
        f"📑 Extracción inteligente: {len(paginas_contenido)} páginas de contenido, "
        f"{n_toc} páginas de índice omitidas, "
        f"{len(estructura_toc)} secciones detectadas en TOC"
    )
    if estructura_toc:
        secciones_str = ', '.join(list(estructura_toc.keys())[:6])
        logger.info(f"Estructura TOC: {secciones_str}{'…' if len(estructura_toc) > 6 else ''}")

    return paginas_contenido, estructura_toc


# ====================================================================== #
#  AGRUPACIÓN POR TOC (portado de langgraph backend/rag/tesis_store.py)   #
# ====================================================================== #
# Chunks por debajo de este tamaño son "solo título" sin cuerpo → se descartan.
_MIN_CHARS_CHUNK = 80


def _encontrar_encabezado_en_texto(texto: str, nombre_seccion: str) -> int:
    """
    Localiza el encabezado de una sección en el texto de una página.

    Returns posición de inicio (0-indexed), o -1 si no se encuentra.
    Cascada: búsqueda exacta → normalización de espacios → prefijo numérico.
    """
    # 1. Búsqueda exacta
    idx = texto.find(nombre_seccion)
    if idx >= 0:
        return idx

    # 2. Normalizar espacios y buscar de nuevo
    nombre_norm = re.sub(r'\s+', ' ', nombre_seccion).strip()
    idx = texto.find(nombre_norm)
    if idx >= 0:
        return idx

    # 3. Buscar por prefijo numérico al inicio de una línea (ej. "1.4.2")
    m_pref = re.match(r'^(\d[\d\.]*)', nombre_norm)
    if m_pref:
        prefix = m_pref.group(1).rstrip('.')
        pattern = r'(?:(?<=\n)|^)' + re.escape(prefix) + r'[.\s]'
        m = re.search(pattern, texto)
        if m:
            pos = m.start()
            return pos + (1 if pos < len(texto) and texto[pos] == '\n' else 0)

    return -1


def _agrupar_por_toc(
    paginas: List[Tuple[int, str]],
    estructura_toc: Dict[str, int],
) -> List[Tuple[str, str, int]]:
    """
    Asigna el texto de cada página a la sección del TOC que le corresponde,
    repartiendo páginas que contienen varios encabezados.

    Returns: lista de (nombre_seccion, texto_agrupado, pagina_inicio).
    """
    if not estructura_toc or not paginas:
        texto_total = "\n\n".join(t for _, t in sorted(paginas))
        return [("Documento completo", texto_total, 1)]

    secciones_ord = sorted(estructura_toc.items(), key=lambda x: x[1])
    acumulado: Dict[str, List[str]] = {nombre: [] for nombre, _ in secciones_ord}
    paginas_asignadas = 0

    for pag, texto_pag in sorted(paginas):
        secciones_en_pag = [n for n, p in secciones_ord if p == pag]

        if not secciones_en_pag:
            # Página de continuación: asignar a la sección en curso (última con inicio ≤ pag)
            running = None
            for nombre, pag_inicio in reversed(secciones_ord):
                if pag_inicio <= pag:
                    running = nombre
                    break
            if running is not None:
                acumulado[running].append(texto_pag)
                paginas_asignadas += 1
        else:
            # Una o más secciones nuevas empiezan en esta página.
            prev = None
            for nombre, pag_inicio in reversed(secciones_ord):
                if pag_inicio < pag:
                    prev = nombre
                    break

            posiciones: Dict[str, int] = {}
            for nombre in secciones_en_pag:
                pos = _encontrar_encabezado_en_texto(texto_pag, nombre)
                if pos >= 0:
                    posiciones[nombre] = pos

            if posiciones:
                secciones_pos = sorted(posiciones.items(), key=lambda x: x[1])
                primera_pos = secciones_pos[0][1]
                if primera_pos > 0 and prev is not None:
                    previo = texto_pag[:primera_pos].strip()
                    if previo:
                        acumulado[prev].append(previo)
                for i, (nombre, pos) in enumerate(secciones_pos):
                    sig = secciones_pos[i + 1][1] if i + 1 < len(secciones_pos) else len(texto_pag)
                    frag = texto_pag[pos:sig].strip()
                    if frag:
                        acumulado[nombre].append(frag)
            else:
                acumulado[secciones_en_pag[-1]].append(texto_pag)

            paginas_asignadas += 1

    if paginas_asignadas == 0:
        logger.warning(
            "TOC detectado pero ninguna página coincide con sus números de página. "
            "Fallback a documento completo."
        )
        texto_total = "\n\n".join(t for _, t in sorted(paginas))
        return [("Documento completo", texto_total, 1)]

    grupos: List[Tuple[str, str, int]] = []
    for nombre, pag_inicio in secciones_ord:
        texto_sec = "\n\n".join(acumulado[nombre])
        if texto_sec.strip():
            grupos.append((nombre, texto_sec.strip(), pag_inicio))

    logger.info(
        f"TOC: {len(grupos)} secciones con contenido "
        f"({paginas_asignadas}/{len(paginas)} páginas asignadas)"
    )
    return grupos


def _section_id_title_level(nombre: str) -> Tuple[str, str, int]:
    """
    Deriva (section_id, title, level) del nombre completo de una sección del TOC.
      "1.1.2 Problema central"  → ("1.1.2", "Problema central", 3)
      "2. MARCO TEÓRICO"        → ("2", "MARCO TEÓRICO", 1)
      "III. Referencias"        → ("III. Referencias", "III. Referencias", 1)
    """
    m = re.match(r'^(\d[\d\.]*)', nombre.strip())
    if m:
        section_id = m.group(1).rstrip('.')
        title = nombre[m.end():].strip().lstrip('.').strip() or nombre.strip()
        level = section_id.count('.') + 1
    else:
        section_id = nombre.strip()
        title = nombre.strip()
        level = 1
    return section_id, title, level


def _secciones_a_chunks(
    grupos: List[Tuple[str, str, int]],
    source_name: str,
    splitter: RecursiveCharacterTextSplitter,
    chunk_size: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convierte los grupos por sección en chunks dict + outline.

    Regla langgraph: si el texto de la sección ≤ chunk_size se guarda como un
    solo chunk; si no, se trocea. Secciones < _MIN_CHARS_CHUNK (solo título)
    se descartan del índice.

    Returns:
        chunks:  [{"text": str, "metadata": {...}}]
        outline: [{section_id, title, page, level, chunks_count, chars_count, seccion}]
    """
    chunks: List[Dict[str, Any]] = []
    outline: List[Dict[str, Any]] = []

    for nombre, texto, pag_inicio in grupos:
        texto_limpio = texto.strip()
        if len(texto_limpio) < _MIN_CHARS_CHUNK:
            logger.debug(f"Sección '{nombre}' descartada ({len(texto_limpio)} chars — solo título)")
            continue

        piezas = [texto_limpio] if len(texto_limpio) <= chunk_size else splitter.split_text(texto_limpio)

        section_id, title, level = _section_id_title_level(nombre)
        chunks_count = 0
        chars_count = 0
        for pieza in piezas:
            t = pieza.strip()
            if not t:
                continue
            idx = len(chunks)
            chunks.append({
                "text": t,
                "metadata": {
                    "source":           source_name,
                    "chunk_id":         f"{source_name}_chunk_{idx:04d}",
                    "seccion":          nombre,
                    "pagina_inicio":    int(pag_inicio),
                    # `page` se conserva (= pagina_inicio) para la viz de embeddings
                    # y el filtro por rango de páginas; el troceo por sección pierde
                    # el page por-chunk fino.
                    "page":             int(pag_inicio),
                    "section_detected": nombre,
                    "char_count":       len(t),
                },
            })
            chunks_count += 1
            chars_count += len(t)

        if chunks_count == 0:
            continue
        outline.append({
            "section_id":   section_id,
            "title":        title,
            "page":         int(pag_inicio),
            "level":        level,
            "chunks_count": chunks_count,
            "chars_count":  chars_count,
            "seccion":      nombre,
        })

    return chunks, outline


def detect_section(text: str) -> str:
    """
    Detecta la sección académica predominante en los primeros 300 caracteres del chunk.
    Retorna 'general' si no coincide con ningún patrón.
    """
    sample = text[:300]
    for pattern, name in _SECTION_PATTERNS:
        if pattern.search(sample):
            return name
    return "general"


def clean_text(text: str) -> str:
    """Limpia el texto extraído del PDF."""
    # Normaliza saltos de línea múltiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Elimina espacios múltiples (pero preserva saltos)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Elimina caracteres de control raros (excepto \n)
    text = re.sub(r"[^\x20-\x7E\n\xC0-\xFFÀ-ɏ]", "", text)
    return text.strip()


def extract_pages(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Extrae texto página a página desde un PDF en bytes.

    Returns:
        Lista de dicts: [{"page": int, "text": str}, ...]
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: List[Dict[str, Any]] = []

    for idx, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        cleaned = clean_text(raw)
        if len(cleaned) > 30:  # ignora páginas casi vacías
            pages.append({"page": idx, "text": cleaned})

    logger.info(f"📄 Páginas con contenido extraídas: {len(pages)} / {len(reader.pages)}")
    return pages


def is_scanned_pdf(
    pdf_bytes: bytes,
    min_chars_per_page: int = 50,
    ratio_threshold: float = 0.9,
) -> bool:
    """
    Heurística para detectar PDFs sin capa de texto (escaneados sin OCR).

    Returns:
        True si al menos `ratio_threshold` (90% por default) de las páginas
        tienen menos de `min_chars_per_page` (50 por default) caracteres
        extraíbles. También True si el PDF está vacío.

    Limitación: no detecta PDFs con OCR de mala calidad (texto basura);
    sólo el caso claro de "no hay texto extraíble".
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        return True  # PDF corrupto

    total = len(reader.pages)
    if total == 0:
        return True

    empty_pages = 0
    for page in reader.pages:
        try:
            text = (page.extract_text() or "").strip()
        except Exception:
            text = ""
        if len(text) < min_chars_per_page:
            empty_pages += 1

    return (empty_pages / total) >= ratio_threshold


def build_chunks(
    pages: List[Dict[str, Any]],
    source_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    """
    Divide las páginas en chunks semánticos y les asigna metadatos.

    Returns:
        Lista de dicts: [{"text": str, "metadata": dict}, ...]
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[Dict[str, Any]] = []

    for page_data in pages:
        page_num = page_data["page"]
        raw_chunks = splitter.split_text(page_data["text"])

        for raw_chunk in raw_chunks:
            text = raw_chunk.strip()
            if len(text) < 50:  # descarta fragmentos demasiado pequeños
                continue

            chunk_idx = len(chunks)
            chunks.append(
                {
                    "text": text,
                    "metadata": {
                        "source": source_name,
                        "page": page_num,
                        "chunk_id": f"{source_name}_chunk_{chunk_idx:04d}",
                        "section_detected": detect_section(text),
                        "char_count": len(text),
                    },
                }
            )

    logger.info(
        f"✂️  Chunking completado: {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def _build_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "   ", " ", ""],
    )


def process_pdf(
    pdf_bytes: bytes,
    filename: str,
) -> Dict[str, Any]:
    """
    Pipeline completo de procesamiento de un PDF de tesis.

    Camino principal (TOC-aware): excluye el índice, agrupa el texto por sección
    del TOC y trocea dentro de cada sección. Si el PDF no tiene índice parseable,
    cae al pipeline clásico página-a-página (keyword + outline jerárquico 1.1.1).

    Returns:
        {
            "filename": str,
            "total_pages": int,
            "pages_with_content": int,
            "chunks": List[{"text": str, "metadata": dict}],
            "sections_found": dict,           # conteo de chunks por sección
            "outline": List[dict],            # secciones para el dropdown del frontend
                                              # con chunks_count y chars_count.
        }
    """
    from app.config import settings

    logger.info(f"🔍 Procesando PDF: {filename}")

    # ── Intento TOC-aware (excluye índice + agrupa por sección) ───────────
    try:
        paginas_contenido, estructura_toc = extraer_contenido_sin_indice(pdf_bytes)
    except Exception as exc:
        logger.warning(f"Extracción TOC-aware falló ({exc}); usando fallback por página.")
        paginas_contenido, estructura_toc = [], {}

    if estructura_toc and paginas_contenido:
        splitter = _build_splitter(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        grupos = _agrupar_por_toc(paginas_contenido, estructura_toc)
        chunks, outline = _secciones_a_chunks(grupos, filename, splitter, settings.CHUNK_SIZE)

        if chunks:
            sections_found: Dict[str, int] = {}
            for chunk in chunks:
                sec = chunk["metadata"]["seccion"]
                sections_found[sec] = sections_found.get(sec, 0) + 1

            total_pages = max((p for p, _ in paginas_contenido), default=0)
            logger.info(
                f"✂️  Chunking por TOC: {len(outline)} secciones → {len(chunks)} chunks"
            )
            return {
                "filename": filename,
                "total_pages": total_pages,
                "pages_with_content": len(paginas_contenido),
                "chunks": chunks,
                "sections_found": sections_found,
                "outline": outline,
            }
        logger.warning("TOC detectado pero no produjo chunks; usando fallback por página.")

    # ── Fallback clásico (PDF sin índice parseable) ───────────────────────
    return _process_pdf_fallback(pdf_bytes, filename, settings)


def _process_pdf_fallback(pdf_bytes: bytes, filename: str, settings) -> Dict[str, Any]:
    """Pipeline clásico página-a-página: keyword + outline jerárquico 1.1.1."""
    logger.info(f"↩️  Fallback página-a-página para: {filename}")

    pages = extract_pages(pdf_bytes)
    chunks = build_chunks(
        pages=pages,
        source_name=filename,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    # Conteo de secciones detectadas (keyword-based)
    sections_found: Dict[str, int] = {}
    for chunk in chunks:
        sec = chunk["metadata"]["section_detected"]
        sections_found[sec] = sections_found.get(sec, 0) + 1

    # Outline jerárquico (1.1.1) — alimenta el dropdown de selección de sección.
    raw_outline = extract_hierarchical_outline(pages)
    outline     = _assign_chunks_to_outline(chunks, raw_outline)

    return {
        "filename": filename,
        "total_pages": len(pages),
        "pages_with_content": len(pages),
        "chunks": chunks,
        "sections_found": sections_found,
        "outline": outline,
    }


def process_reference_pdf(
    pdf_bytes: bytes,
    filename: str,
) -> Dict[str, Any]:
    """
    Pipeline de procesamiento para libros de la Biblioteca Metodológica.

    Equivalente a `agregar_libro` de langgraph: excluye el índice y trocea por
    página de contenido (los libros se recuperan semánticamente, no requieren
    agrupación por sección de tesis). Cae al pipeline clásico si pdfplumber falla.

    Returns: la misma forma que process_pdf.
    """
    from app.config import settings

    logger.info(f"📚 Procesando libro de referencia: {filename}")

    try:
        paginas_contenido, _ = extraer_contenido_sin_indice(pdf_bytes)
    except Exception as exc:
        logger.warning(f"Extracción TOC-aware del libro falló ({exc}); usando fallback.")
        paginas_contenido = []

    if not paginas_contenido:
        return _process_pdf_fallback(pdf_bytes, filename, settings)

    splitter = _build_splitter(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    chunks: List[Dict[str, Any]] = []
    for page_num, texto in paginas_contenido:
        for pieza in splitter.split_text(texto):
            t = pieza.strip()
            if len(t) < 50:   # descarta fragmentos demasiado pequeños
                continue
            idx = len(chunks)
            chunks.append({
                "text": t,
                "metadata": {
                    "source":           filename,
                    "chunk_id":         f"{filename}_chunk_{idx:04d}",
                    "tipo":             "libro_metodologia",
                    "page":             int(page_num),
                    "section_detected": "libro_metodologia",
                    "char_count":       len(t),
                },
            })

    total_pages = max((p for p, _ in paginas_contenido), default=0)
    logger.info(f"✂️  Libro '{filename}' troceado: {len(chunks)} fragmentos (índice excluido)")
    return {
        "filename": filename,
        "total_pages": total_pages,
        "pages_with_content": len(paginas_contenido),
        "chunks": chunks,
        "sections_found": {"libro_metodologia": len(chunks)},
        "outline": [],
    }
