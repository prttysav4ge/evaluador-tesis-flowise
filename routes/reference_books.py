"""
Ruta: GET /api/v1/reference-books

Lista los libros metodológicos de referencia indexados en la colección
'reference_books'. Consumido por el sidebar Streamlit ('Biblioteca
Metodológica') para mostrar los libros disponibles + sus fragmentos.

La población inicial de esta colección se hace UNA SOLA VEZ con el script
scripts/index_reference_books.py.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter

from vectorstore.refs_store import refs_store

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/reference-books", summary="Lista libros metodológicos indexados")
async def list_reference_books() -> Dict[str, Any]:
    """
    Devuelve el catálogo de libros metodológicos en la Biblioteca.

    Returns:
        {
            "books": [{"source", "title", "fragments"}, ...],
            "total_books": int,
            "total_fragments": int,
        }
    """
    books = refs_store.list_books()
    return {
        "books":           books,
        "total_books":     len(books),
        "total_fragments": sum(b["fragments"] for b in books),
    }
