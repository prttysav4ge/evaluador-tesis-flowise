"""
Capa de abstracción sobre ChromaDB.

Responsabilidades:
  - Inicializar el cliente persistente
  - Agregar chunks con sus embeddings y metadatos
  - Consultar por similitud semántica
  - Exponer información de la colección
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)


class ChromaStore:
    """
    Wrapper singleton sobre ChromaDB con API simplificada.
    Se inicializa con initialize() al arrancar el servidor.
    """

    _instance: "ChromaStore | None" = None
    _client: chromadb.PersistentClient | None = None
    _collection: chromadb.Collection | None = None

    def __new__(cls) -> "ChromaStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # ------------------------------------------------------------------ #
    #  Inicialización                                                      #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """Debe llamarse una vez al iniciar el servidor (lifespan)."""
        from app.config import settings

        self._client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        total = self._collection.count()
        logger.info(
            f"✅ ChromaDB listo | Colección: '{settings.CHROMA_COLLECTION}' "
            f"| Chunks almacenados: {total} "
            f"| Directorio: {settings.CHROMA_PERSIST_DIR}"
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
        ids: Optional[List[str]] = None,
    ) -> int:
        """
        Agrega documentos a ChromaDB.
        Genera los embeddings internamente via el embedder singleton.

        Returns:
            Cantidad de documentos agregados.
        """
        from embeddings.embedder import embedder

        if not texts:
            return 0

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        logger.info(f"⏳ Generando embeddings para {len(texts)} chunks…")
        embeddings = embedder.embed_documents(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"✅ {len(texts)} chunks almacenados en ChromaDB")
        return len(texts)

    # ------------------------------------------------------------------ #
    #  Lectura / Retrieval                                                 #
    # ------------------------------------------------------------------ #

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Consulta semántica contra ChromaDB.

        Returns:
            Lista de dicts con keys: text, metadata, score (distancia coseno, menor = más similar).
        """
        from embeddings.embedder import embedder

        query_embedding = embedder.embed_query(query_text)

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self.collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        raw = self.collection.query(**query_kwargs)

        results: List[Dict[str, Any]] = []
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            results.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "score": round(float(dist), 4),
                }
            )

        return results

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Convierte los resultados de una query en un bloque de texto
        listo para incluir en un prompt.
        """
        if not results:
            return "No se encontraron fragmentos relevantes en la tesis."

        parts: List[str] = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"]
            parts.append(
                f"[Fragmento {i} | Página {meta.get('page', '?')} "
                f"| Sección: {meta.get('section_detected', 'general')}]\n"
                f"{r['text']}"
            )
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------ #
    #  Administración                                                      #
    # ------------------------------------------------------------------ #

    def get_info(self) -> Dict[str, Any]:
        """Retorna estadísticas de la colección actual."""
        from app.config import settings

        count = self.collection.count()
        return {
            "collection": settings.CHROMA_COLLECTION,
            "total_chunks": count,
            "persist_dir": settings.CHROMA_PERSIST_DIR,
            "status": "ready" if count > 0 else "empty",
        }

    def reset(self) -> bool:
        """Elimina y recrea la colección (borra todos los chunks)."""
        from app.config import settings

        self._client.delete_collection(settings.CHROMA_COLLECTION)
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("⚠️  Colección ChromaDB reiniciada (todos los datos borrados)")
        return True


# Instancia singleton
chroma_store = ChromaStore()
