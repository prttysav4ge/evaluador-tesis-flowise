"""
Embedder usando intfloat/multilingual-e5-small.

IMPORTANTE sobre el modelo multilingual-e5:
  - Documentos (indexar): se prefixa con "passage: "
  - Consultas    (buscar): se prefixa con "query: "

El incumplimiento de este convenio degrada severamente la calidad del retrieval.
"""
from __future__ import annotations

import logging
from typing import List

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MultilingualE5Embedder:
    """
    Singleton embedder con el modelo multilingual-e5-small.
    Se carga una sola vez en memoria para toda la vida del servidor.
    """

    _instance: "MultilingualE5Embedder | None" = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> "MultilingualE5Embedder":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_model(self) -> None:
        if self._model is None:
            from app.config import settings
            model_name = settings.EMBEDDING_MODEL
            logger.info(f"⏳ Cargando modelo de embeddings: {model_name}")
            self._model = SentenceTransformer(model_name)
            logger.info(f"✅ Modelo cargado: {model_name}")

    # ------------------------------------------------------------------ #
    #  API pública                                                         #
    # ------------------------------------------------------------------ #

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos de tipo 'pasaje' (documentos).
        Agrega el prefijo 'passage: ' requerido por multilingual-e5.
        """
        self._load_model()
        prefixed = [f"passage: {t.strip()}" for t in texts]
        vectors = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> List[float]:
        """
        Genera embedding para una consulta.
        Agrega el prefijo 'query: ' requerido por multilingual-e5.
        """
        self._load_model()
        prefixed = f"query: {query.strip()}"
        vector = self._model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vector[0].tolist()

    @property
    def dimension(self) -> int:
        """Dimensión del vector de embedding del modelo."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()


# Instancia singleton — importar desde aquí en el resto del proyecto
embedder = MultilingualE5Embedder()
