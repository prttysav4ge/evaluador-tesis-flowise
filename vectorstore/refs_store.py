"""
Store paralelo a ChromaStore para los libros metodológicos de referencia
(Hernández Sampieri, Tamayo, etc.). Vive en la MISMA instancia de
PersistentClient que ChromaStore pero en una colección aparte
('reference_books') para no mezclar fragmentos de la tesis del estudiante
con fragmentos de los libros de consulta.

Usado por:
  - scripts/index_reference_books.py — indexa los PDFs una sola vez.
  - routes/reference_books.py        — endpoint que alimenta el sidebar.
  - routes/query.py                  — retrieval cruzado tesis + libros.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)

REFS_COLLECTION_NAME = "reference_books"


def _filename_to_title(filename: str) -> str:
    """
    Normaliza un nombre de archivo PDF a un título legible:
    quita la extensión y normaliza espacios.
    """
    name = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    return name.strip()


class RefsStore:
    """Singleton para la colección Chroma de libros metodológicos."""

    _instance: "RefsStore | None"            = None
    _collection: chromadb.Collection | None  = None

    def __new__(cls) -> "RefsStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------ #
    #  Inicialización                                                      #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """
        Conecta a ChromaDB y crea/abre la colección de refs.
        Reusa el PersistentClient de ChromaStore para no mantener dos
        clientes apuntando al mismo persist_dir.
        """
        from vectorstore.chroma_store import chroma_store

        if chroma_store._client is None:
            chroma_store.initialize()

        self._collection = chroma_store._client.get_or_create_collection(
            name=REFS_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        total = self._collection.count()
        logger.info(
            f"📚 RefsStore listo | Colección: '{REFS_COLLECTION_NAME}' "
            f"| Chunks: {total}"
        )

    @property
    def collection(self) -> chromadb.Collection:
        if self._collection is None:
            self.initialize()
        return self._collection

    # ------------------------------------------------------------------ #
    #  Escritura                                                           #
    # ------------------------------------------------------------------ #

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
    ) -> int:
        """Agrega chunks con sus embeddings. Devuelve cuántos se agregaron."""
        from embeddings.embedder import embedder

        if not texts:
            return 0

        logger.info(f"⏳ Generando embeddings para {len(texts)} chunks de refs…")
        embeddings = embedder.embed_documents(texts)
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"✅ {len(texts)} chunks de refs almacenados")
        return len(texts)

    # ------------------------------------------------------------------ #
    #  Lectura                                                             #
    # ------------------------------------------------------------------ #

    def list_books(self) -> List[Dict[str, Any]]:
        """
        Devuelve la lista de libros indexados agregando por metadata['source'].

        Returns:
            [{"source": "...", "title": "...", "fragments": N}, ...]
        """
        try:
            all_data = self.collection.get(include=["metadatas"])
        except Exception as exc:
            logger.warning(f"list_books falló: {exc}")
            return []

        metas: List[Dict[str, Any]] = all_data.get("metadatas") or []
        counts: Dict[str, int] = {}
        for m in metas:
            src = (m or {}).get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1

        return [
            {
                "source":    src,
                "title":     _filename_to_title(src),
                "fragments": n,
            }
            for src, n in sorted(counts.items())
        ]

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieval semántico contra refs. Si se pasa source_filter, restringe
        la búsqueda a un libro específico.

        Returns:
            Lista de dicts {text, metadata, score}.
        """
        from embeddings.embedder import embedder

        query_embedding = embedder.embed_query(query_text)
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if source_filter:
            kwargs["where"] = {"source": source_filter}

        raw = self.collection.query(**kwargs)
        docs  = raw.get("documents",  [[]])[0]
        metas = raw.get("metadatas",  [[]])[0]
        dists = raw.get("distances",  [[]])[0]

        return [
            {"text": d, "metadata": m, "score": round(float(s), 4)}
            for d, m, s in zip(docs, metas, dists)
        ]

    def reset(self) -> bool:
        """Elimina y recrea la colección de refs (borra todos los chunks)."""
        from vectorstore.chroma_store import chroma_store

        chroma_store._client.delete_collection(REFS_COLLECTION_NAME)
        self._collection = chroma_store._client.get_or_create_collection(
            name=REFS_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("⚠️  Colección reference_books reiniciada")
        return True


def index_reference_books(pdf_dir: str = "reference_books") -> int:
    """
    Indexa todos los PDFs de `pdf_dir` en la colección reference_books.
    Idempotente: skip los archivos cuyo `source` ya está en la colección.

    Usado por:
      - scripts/index_reference_books.py (manual, ejecutable desde CLI).
      - main.py lifespan en producción (auto-index al startup si la
        colección Chroma está vacía — necesario en Streamlit Cloud donde
        no podés correr scripts manualmente).

    Returns:
        Cantidad de chunks nuevos agregados (0 si todo ya estaba indexado).
    """
    from pathlib import Path
    from services.pdf_service import process_reference_pdf

    refs_store.initialize()

    base = Path(pdf_dir)
    if not base.is_dir():
        logger.warning(
            f"index_reference_books: carpeta '{pdf_dir}' no existe. "
            "Saltando indexación."
        )
        return 0

    pdfs = sorted(base.glob("*.pdf"))
    if not pdfs:
        logger.warning(
            f"index_reference_books: '{pdf_dir}' no tiene PDFs. "
            "Saltando indexación."
        )
        return 0

    existing_sources = {b["source"] for b in refs_store.list_books()}
    total_added = 0
    for pdf_path in pdfs:
        filename = pdf_path.name
        if filename in existing_sources:
            logger.info(f"   ✔ skip (ya indexado): {filename}")
            continue

        logger.info(f"   📄 indexando: {filename}")
        try:
            result = process_reference_pdf(pdf_path.read_bytes(), filename)
        except Exception as exc:
            logger.error(f"      ❌ error procesando: {exc}")
            continue

        chunks = result["chunks"]
        if not chunks:
            logger.warning("      ⚠️ sin chunks extraídos, saltando")
            continue

        texts     = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids       = [f"refs::{filename}::{i:04d}" for i in range(len(chunks))]
        n = refs_store.add_documents(texts, metadatas, ids)
        total_added += n
        logger.info(f"      ✅ {n} chunks indexados")

    return total_added


# Singleton importable
refs_store = RefsStore()
