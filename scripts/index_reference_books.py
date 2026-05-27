"""
Script de indexación de la Biblioteca Metodológica.

Lee los PDFs metodológicos de referencia y los almacena en la colección
ChromaDB 'reference_books' (separada de la colección de tesis). Esta
indexación se hace UNA SOLA VEZ por instalación; los chunks persisten en
./chroma_db/ entre ejecuciones de la app.

Uso:
    python scripts/index_reference_books.py [path_a_carpeta_pdfs]

Por defecto busca PDFs en:
    1. ./reference_books/  (si existe — recomendado)
    2. ./                   (fallback al root del repo)

Los chunks ya indexados no se re-procesan (idempotente por filename).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Forzar UTF-8 para los emojis en consola Windows (cp1252 por default).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Permite ejecutar desde la raíz del repo: python scripts/index_reference_books.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Bootstrap de secrets (igual que streamlit_app.py) para que app.config tenga
# las vars de entorno antes de que se importen los servicios.
try:
    import tomllib  # py3.11+
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        with open(secrets_path, "rb") as f:
            for k, v in tomllib.load(f).items():
                if isinstance(v, (str, int, float, bool)):
                    os.environ.setdefault(k, str(v))
except Exception:
    pass


# Nombres conocidos de los 4 PDFs metodológicos. Si tu carpeta tiene otros,
# el script igual los indexará (escanea *.pdf en la carpeta elegida).
_KNOWN_PDFS = [
    "METODOLOGIA DE LA INVESTIGACION CUANTITATIVA-CUALITATIVA Y REDACCION DE LA TESIS.pdf",
    "METODOLOGIA DE LA INVESTIGACION-GUIA PARA EL PROYECTO DE TESIS.pdf",
    "METODOLOGÍA DE LA INVESTIGACION-LAS RUTAS CUANTITATIVA, CUALITATIVA Y MIXTA.pdf",
    "PROYECTO DE TESIS-GUIA PRACTICA PARA INVESTIGACION CUANTITATIVA.pdf",
]


def main() -> int:
    from services.pdf_service import process_pdf
    from vectorstore.chroma_store import chroma_store
    from vectorstore.refs_store   import refs_store

    # ── Resolver carpeta de PDFs ─────────────────────────────────────────
    if len(sys.argv) > 1:
        pdf_dir = Path(sys.argv[1])
    elif Path("reference_books").is_dir():
        pdf_dir = Path("reference_books")
    else:
        pdf_dir = Path(".")
    print(f"📂 Carpeta de PDFs: {pdf_dir.resolve()}")

    # ── Encontrar PDFs ───────────────────────────────────────────────────
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print("❌ No se encontraron PDFs en la carpeta. Sal.")
        return 1
    print(f"📚 PDFs encontrados: {len(pdfs)}")

    # ── Inicializar stores ───────────────────────────────────────────────
    chroma_store.initialize()
    refs_store.initialize()

    # ── Saltar los ya indexados ──────────────────────────────────────────
    existing       = refs_store.list_books()
    existing_srcs  = {b["source"] for b in existing}
    if existing_srcs:
        print(f"   {len(existing_srcs)} ya indexado(s): {sorted(existing_srcs)}")

    # ── Indexar uno por uno ──────────────────────────────────────────────
    for pdf_path in pdfs:
        filename = pdf_path.name
        if filename in existing_srcs:
            print(f"✔ Skip (ya indexado): {filename}")
            continue

        print(f"📄 Procesando: {filename}")
        pdf_bytes = pdf_path.read_bytes()
        try:
            result = process_pdf(pdf_bytes, filename)
        except Exception as exc:
            print(f"   ❌ Error procesando: {exc}")
            continue

        chunks = result["chunks"]
        if not chunks:
            print("   ⚠️ Sin chunks extraídos. Sal.")
            continue

        texts     = [c["text"]     for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids       = [f"refs::{filename}::{i:04d}" for i in range(len(chunks))]
        n         = refs_store.add_documents(texts, metadatas, ids)
        print(f"   ✅ {n} chunks indexados")

    # ── Resumen ──────────────────────────────────────────────────────────
    print()
    print("📚 Biblioteca Metodológica:")
    books = refs_store.list_books()
    for b in books:
        print(f"   📖 {b['title'][:70]:<70}  {b['fragments']:>6} frags")
    total = sum(b["fragments"] for b in books)
    print(f"   ── Total: {len(books)} libro(s) · {total} fragmentos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
