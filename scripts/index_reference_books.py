"""
Script de indexación de la Biblioteca Metodológica.

Wrapper sobre vectorstore.refs_store.index_reference_books() para ejecutar
la indexación manualmente desde la CLI (útil en desarrollo local). En
producción no es necesario correr esto: el lifespan de main.py llama a la
misma función automáticamente si la colección Chroma está vacía.

Uso:
    python scripts/index_reference_books.py [path_a_carpeta_pdfs]

Por defecto busca PDFs en ./reference_books/ (carpeta trackeada en el repo).
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


def main() -> int:
    from vectorstore.chroma_store import chroma_store
    from vectorstore.refs_store   import refs_store, index_reference_books

    pdf_dir = sys.argv[1] if len(sys.argv) > 1 else "reference_books"
    print(f"📂 Carpeta de PDFs: {Path(pdf_dir).resolve()}")

    chroma_store.initialize()
    refs_store.initialize()

    added = index_reference_books(pdf_dir)
    print()
    print(f"📚 Biblioteca Metodológica (chunks agregados en esta corrida: {added}):")
    books = refs_store.list_books()
    for b in books:
        print(f"   📖 {b['title'][:70]:<70}  {b['fragments']:>6} frags")
    total = sum(b["fragments"] for b in books)
    print(f"   ── Total: {len(books)} libro(s) · {total} fragmentos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
