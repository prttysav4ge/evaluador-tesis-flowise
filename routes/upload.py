"""
Ruta: POST /api/v1/upload-pdf

Recibe un PDF, lo procesa (extracción → chunking → embeddings) y
almacena los chunks en ChromaDB.
"""
from __future__ import annotations

import logging
from typing import Dict, Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from services.pdf_service import is_scanned_pdf, process_pdf
from vectorstore.chroma_store import chroma_store

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_FILE_SIZE_MB = 50
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}


@router.post(
    "/upload-pdf",
    summary="Subir y procesar un PDF de tesis",
    response_description="Estadísticas del procesamiento y almacenamiento",
)
async def upload_pdf(
    file: UploadFile = File(..., description="Archivo PDF de la tesis"),
) -> Dict[str, Any]:
    """
    Pipeline completo de ingesta de un PDF:

    1. Validación del archivo
    2. Extracción de texto (pypdf)
    3. Chunking semántico (RecursiveCharacterTextSplitter)
    4. Generación de embeddings (multilingual-e5-small)
    5. Almacenamiento en ChromaDB

    Después de subir el PDF puedes hacer consultas en **POST /api/v1/query**.
    """
    # ------------------------------------------------------------------ #
    #  Validaciones                                                        #
    # ------------------------------------------------------------------ #
    if file.content_type not in ALLOWED_CONTENT_TYPES and not (
        file.filename and file.filename.lower().endswith(".pdf")
    ):
        raise HTTPException(
            status_code=400,
            detail="Solo se aceptan archivos PDF.",
        )

    pdf_bytes = await file.read()

    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"El archivo supera el límite de {MAX_FILE_SIZE_MB} MB. "
                   f"Tamaño recibido: {size_mb:.1f} MB.",
        )

    if len(pdf_bytes) < 100:
        raise HTTPException(status_code=400, detail="El PDF parece estar vacío.")

    filename = file.filename or "tesis_sin_nombre.pdf"

    # ------------------------------------------------------------------ #
    #  Validación: PDF escaneado sin OCR                                    #
    # ------------------------------------------------------------------ #
    # Hacemos esta verificación ANTES del procesamiento completo para
    # rechazar rápido y con mensaje claro, en vez de fallar más adelante
    # con un 422 ambiguo al no encontrar chunks.
    if is_scanned_pdf(pdf_bytes):
        raise HTTPException(
            status_code=400,
            detail=(
                "No se puede leer este PDF — parece escaneado sin OCR. "
                "Convierte el PDF a texto (con un OCR como Adobe Acrobat, "
                "Tesseract o ABBYY FineReader) antes de subirlo."
            ),
        )

    # ------------------------------------------------------------------ #
    #  Procesamiento                                                       #
    # ------------------------------------------------------------------ #
    logger.info(f"📥 PDF recibido: '{filename}' ({size_mb:.2f} MB)")

    try:
        result = process_pdf(pdf_bytes, filename)
    except Exception as exc:
        logger.exception(f"Error al procesar el PDF '{filename}'")
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar el PDF: {str(exc)}",
        )

    chunks = result["chunks"]
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No se pudo extraer texto del PDF. "
                   "Verifica que no sea un PDF escaneado (imagen).",
        )

    # ------------------------------------------------------------------ #
    #  Almacenamiento en ChromaDB                                          #
    # ------------------------------------------------------------------ #
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    # IDs únicos por source + chunk_id para evitar duplicados
    ids = [c["metadata"]["chunk_id"] for c in chunks]

    try:
        stored = chroma_store.add_documents(texts, metadatas, ids=ids)
    except Exception as exc:
        logger.exception("Error al guardar en ChromaDB")
        raise HTTPException(
            status_code=500,
            detail=f"Error al almacenar en ChromaDB: {str(exc)}",
        )

    # ------------------------------------------------------------------ #
    #  Respuesta                                                           #
    # ------------------------------------------------------------------ #
    return {
        "success": True,
        "filename": filename,
        "file_size_mb": round(size_mb, 2),
        "total_pages": result["total_pages"],
        "chunks_generated": len(chunks),
        "chunks_stored": stored,
        "sections_found": result["sections_found"],
        # Outline jerárquico (1.1.1) con chunks_count y chars_count por sección.
        # Vacío si el PDF no usa numeración; el frontend cae a sections_found.
        "outline": result.get("outline", []),
        "message": (
            f"✅ PDF procesado correctamente. "
            f"{stored} fragmentos almacenados en ChromaDB. "
            f"Ya puedes hacer consultas en POST /api/v1/query."
        ),
    }
