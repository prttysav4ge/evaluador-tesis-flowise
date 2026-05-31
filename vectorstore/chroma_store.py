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
import re
import uuid
from typing import Any, Dict, List, Optional

import chromadb

logger = logging.getLogger(__name__)

# Límite de fragmentos por sección (~12 000 chars máx), igual que langgraph.
MAX_FRAGMENTOS_SECCION = 20


# ── Helpers de jerarquía de secciones (portados de langgraph tesis_store.py) ──

def _extraer_prefijo(nombre: str) -> str:
    """Extrae el prefijo numérico de una sección: '2.1. Título' → '2.1'."""
    m = re.match(r'^(\d[\d\.]*)', nombre.strip())
    return m.group(1).rstrip('.') if m else ""


def _es_subseccion(nombre: str, prefijo_padre: str) -> bool:
    """True si la sección pertenece al prefijo padre o es subsección de él."""
    if not prefijo_padre:
        return False
    p = _extraer_prefijo(nombre)
    return p == prefijo_padre or p.startswith(prefijo_padre + ".")


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

    def query_by_section(
        self,
        seccion: str,
        k: int = MAX_FRAGMENTOS_SECCION,
        fallback_question: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Recupera los chunks de una sección (y sus subsecciones) por metadata
        `seccion`, ordenados por `pagina_inicio`. Replica la intención de
        `recuperar_contexto` de langgraph: acotar el contexto a la sección
        elegida en lugar de una búsqueda semántica global.

        Si ningún chunk coincide por metadata (p.ej. PDF sin TOC o nombre
        distinto), cae a una búsqueda semántica global con `fallback_question`.

        Returns:
            Lista de dicts {text, metadata, score} (score=None en match directo).
        """
        try:
            data = self.collection.get(include=["metadatas", "documents"])
        except Exception as exc:
            logger.error(f"query_by_section: collection.get falló: {exc}")
            data = {}

        metas = data.get("metadatas") or []
        docs  = data.get("documents") or []

        prefijo = _extraer_prefijo(seccion)
        matched: List[Dict[str, Any]] = []
        for doc, meta in zip(docs, metas):
            sec_meta = (meta or {}).get("seccion", "")
            if not sec_meta:
                continue
            if sec_meta == seccion or _es_subseccion(sec_meta, prefijo):
                matched.append({"text": doc, "metadata": meta, "score": None})

        if matched:
            matched.sort(key=lambda r: r["metadata"].get("pagina_inicio", 0))
            logger.info(
                f"🔎 query_by_section('{seccion}'): {len(matched)} fragmentos por metadata "
                f"(cap {k})"
            )
            return matched[:k]

        # Fallback: búsqueda semántica global
        logger.info(
            f"🔎 query_by_section('{seccion}') sin match por metadata; "
            "fallback a búsqueda semántica global."
        )
        return self.query(fallback_question or seccion, top_k=k)

    def format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Convierte los resultados de una query en un bloque de texto
        listo para incluir en un prompt.
        """
        if not results:
            return "No se encontraron fragmentos relevantes en la tesis."

        parts: List[str] = []
        for i, r in enumerate(results, 1):
            meta = r["metadata"] or {}
            # Preferimos el nombre real de la sección del TOC; si no existe
            # (fallback/refs), caemos a section_detected.
            seccion = meta.get("seccion") or meta.get("section_detected", "general")
            parts.append(
                f"[Fragmento {i} | Página {meta.get('page', '?')} "
                f"| Sección: {seccion}]\n"
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
