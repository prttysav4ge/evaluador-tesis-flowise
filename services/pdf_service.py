"""
Servicio de procesamiento de PDFs.

Pipeline:
  1. Extrae texto página a página con pypdf
  2. Limpia el texto (saltos de línea excesivos, espacios, etc.)
  3. Divide en chunks semánticos con RecursiveCharacterTextSplitter
  4. Detecta la sección académica de cada chunk (heurística por regex)
  5. Retorna chunks con metadatos listos para guardar en ChromaDB
"""
from __future__ import annotations

import io
import re
import logging
from typing import Any, Dict, List, Tuple

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
# Encabezados de sección con numeración tipo "1.", "1.1", "1.1.1.", hasta 4 niveles.
# El título debe empezar con mayúscula (latina o acentuada) y tener 3-100 chars.
# Se aplica con re.MULTILINE sobre el texto limpio de cada página.
_HIERARCHICAL_HEADING_RE = re.compile(
    r"^[ \t]*(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-ZÁÉÍÓÚÑa-zá-úñ][^\n]{2,99})[ \t]*$",
    re.MULTILINE,
)


def _looks_like_bibliography_entry(title: str) -> bool:
    """
    Filtro heurístico: descarta líneas que parecen citas bibliográficas
    (autores con año entre paréntesis) en lugar de títulos de sección.
    """
    return bool(re.search(r"\(\s*(?:19|20)\d{2}", title))


def extract_hierarchical_outline(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detecta encabezados jerárquicos (1.1.1) en el texto de cada página y
    construye un outline ordenado en aparición.

    Returns:
        [{"section_id": "1.1.1", "title": "Antecedentes", "page": 12, "level": 3}, ...]

    Si el PDF no usa numeración (ej. solo keywords), retorna lista vacía;
    el caller puede entonces caer a la detección por keyword.
    """
    outline: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for page_data in pages:
        page_num = page_data["page"]
        text     = page_data["text"]

        for match in _HIERARCHICAL_HEADING_RE.finditer(text):
            section_id = match.group(1).rstrip(".")
            title      = match.group(2).strip()

            if _looks_like_bibliography_entry(title):
                continue
            # Evitar duplicados por re-detección del mismo heading en distintas páginas
            # (puede pasar si el heading aparece en un índice y luego en el cuerpo).
            # Conservamos la PRIMERA aparición (la del índice o el body, lo que venga antes).
            if section_id in seen_ids:
                continue
            seen_ids.add(section_id)

            outline.append({
                "section_id": section_id,
                "title":      title,
                "page":       page_num,
                "level":      section_id.count(".") + 1,
            })

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


def process_pdf(
    pdf_bytes: bytes,
    filename: str,
) -> Dict[str, Any]:
    """
    Pipeline completo de procesamiento de un PDF.

    Returns:
        {
            "filename": str,
            "total_pages": int,
            "pages_with_content": int,
            "chunks": List[{"text": str, "metadata": dict}],
            "sections_found": dict,           # conteo keyword-based (legacy)
            "outline": List[dict],            # encabezados jerárquicos (1.1.1)
                                              # con chunks_count y chars_count.
                                              # Vacío si el PDF no usa numeración.
        }
    """
    from app.config import settings

    logger.info(f"🔍 Procesando PDF: {filename}")

    pages = extract_pages(pdf_bytes)
    chunks = build_chunks(
        pages=pages,
        source_name=filename,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    # Conteo de secciones detectadas (keyword-based, legacy)
    sections_found: Dict[str, int] = {}
    for chunk in chunks:
        sec = chunk["metadata"]["section_detected"]
        sections_found[sec] = sections_found.get(sec, 0) + 1

    # Outline jerárquico (1.1.1) — alimenta el dropdown del Sprint 3.
    # _assign_chunks_to_outline mutará chunks[*]['metadata']['outline_section_id'].
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
