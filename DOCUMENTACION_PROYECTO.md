# Documentación Técnica del Proyecto
# POC: Evaluador de Tesis Universitarias — Sistema RAG Multiagente

**Versión:** 1.0.0  
**Fecha:** Mayo 2026  
**Tipo:** Prueba de Concepto (Proof of Concept)

---

## Tabla de Contenidos

1. [Descripción General](#1-descripción-general)
2. [Arquitectura del Sistema](#2-arquitectura-del-sistema)
3. [Stack Tecnológico](#3-stack-tecnológico)
4. [Estructura del Proyecto](#4-estructura-del-proyecto)
5. [Módulos y Componentes](#5-módulos-y-componentes)
   - 5.1 [Configuración Central](#51-configuración-central)
   - 5.2 [Embeddings](#52-módulo-de-embeddings)
   - 5.3 [Vector Store (ChromaDB)](#53-vector-store-chromadb)
   - 5.4 [Servicio de PDFs](#54-servicio-de-procesamiento-de-pdfs)
   - 5.5 [Pipeline de Agentes](#55-pipeline-de-agentes-secuenciales)
   - 5.6 [Cliente Flowise](#56-cliente-flowise)
   - 5.7 [API REST (Rutas)](#57-api-rest)
   - 5.8 [Interfaz Visual (Streamlit)](#58-interfaz-visual-streamlit)
6. [Los 6 Agentes Especializados](#6-los-6-agentes-especializados)
7. [Flujo de Datos Completo](#7-flujo-de-datos-completo)
8. [API Reference](#8-api-reference)
9. [Variables de Entorno](#9-variables-de-entorno)
10. [Modos de Ejecución](#10-modos-de-ejecución)
11. [Instrucciones de Instalación y Ejecución](#11-instrucciones-de-instalación-y-ejecución)
12. [Diagrama de Flujo de Agentes](#12-diagrama-de-flujo-de-agentes)

---

## 1. Descripción General

Este proyecto es un **sistema de evaluación académica automatizada** de proyectos de investigación y tesis universitarias, basado en la técnica **RAG (Retrieval-Augmented Generation)** y un pipeline de **múltiples agentes LLM especializados** que trabajan de forma secuencial y acumulativa.

### Objetivo

Permitir a docentes, tutores y estudiantes obtener una evaluación profunda, estructurada y pedagógica de los documentos académicos, cubriendo dimensiones de:

- Calidad investigativa
- Rigor académico
- Marco metodológico
- Calidad de redacción
- Síntesis pedagógica y recomendaciones priorizadas

### Capacidades principales

| Capacidad | Descripción |
|-----------|-------------|
| **Ingesta de PDF** | Carga, extracción y segmentación automática del documento académico |
| **Búsqueda semántica** | Recuperación de fragmentos relevantes por similitud vectorial (RAG) |
| **Evaluación multiagente** | 6 agentes especializados en cadena con memoria acumulativa |
| **Texto sugerido** | Generación de versión mejorada de la sección analizada |
| **Interfaz visual** | Dashboard Streamlit con visualizaciones de embeddings y evaluaciones |
| **Doble modo** | Agentes en Flowise (producción) o en Python puro (testing) |

---

## 2. Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CAPA DE PRESENTACIÓN                         │
│                                                                     │
│   ┌──────────────────────┐      ┌────────────────────────────────┐  │
│   │  Streamlit App       │      │  Clientes HTTP externos        │  │
│   │  (localhost:8501)    │      │  (curl, Postman, etc.)         │  │
│   └──────────┬───────────┘      └───────────────┬────────────────┘  │
└──────────────┼───────────────────────────────────┼──────────────────┘
               │ HTTP                              │ HTTP
               ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        CAPA DE API (FastAPI)                        │
│                        localhost:8000                               │
│                                                                     │
│   POST /api/v1/upload-pdf   POST /api/v1/query                      │
│   GET  /api/v1/health       GET  /api/v1/collection                 │
│   DELETE /api/v1/collection GET  /api/v1/chunks                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┼─────────────────────┐
              │                │                     │
              ▼                ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐
│ PDF Service     │  │ RAG Retrieval   │  │ Agent Pipeline          │
│                 │  │                 │  │                         │
│ - Extracción    │  │ ChromaDB        │  │ Modo A: Flowise         │
│ - Limpieza      │  │ (vectores cos.) │  │   - Agentflow HTTP      │
│ - Chunking      │  │                 │  │   - 6 nodos LLM         │
│ - Detección     │  │ multilingual-   │  │                         │
│   de secciones  │  │ e5-small        │  │ Modo B: Python puro     │
└────────┬────────┘  └────────┬────────┘  │   - LangChain           │
         │                   │            │   - OpenAI/Groq/Ollama  │
         └──────────┬─────────┘            └──────────┬──────────────┘
                    │                                  │
                    ▼                                  ▼
            ┌───────────────┐                ┌────────────────────┐
            │  ChromaDB     │                │  Texto Sugerido    │
            │  Persistente  │                │  (post-pipeline)   │
            │  ./chroma_db  │                │  Groq/OpenAI/Ollama│
            └───────────────┘                └────────────────────┘
```

---

## 3. Stack Tecnológico

### Backend

| Tecnología | Versión | Rol |
|-----------|---------|-----|
| **Python** | 3.14 | Lenguaje principal |
| **FastAPI** | ≥ 0.116.0 | Framework web / API REST |
| **Uvicorn** | ≥ 0.29.0 | Servidor ASGI |
| **Pydantic** | ≥ 2.5.0 | Validación de datos y settings |
| **LangChain** | ≥ 0.3.0 | Orquestación de LLMs |
| **langchain-openai** | ≥ 0.2.0 | Integración con OpenAI/Groq |
| **httpx** | ≥ 0.27.0 | Cliente HTTP asíncrono (Flowise) |

### Vectores y Embeddings

| Tecnología | Versión | Rol |
|-----------|---------|-----|
| **ChromaDB** | ≥ 0.5.0 | Base de datos vectorial persistente |
| **sentence-transformers** | ≥ 2.7.0 | Generación de embeddings |
| **intfloat/multilingual-e5-small** | — | Modelo de embeddings multilingüe |

### Procesamiento de documentos

| Tecnología | Versión | Rol |
|-----------|---------|-----|
| **pypdf** | ≥ 4.0.0 | Extracción de texto de PDFs |
| **langchain-text-splitters** | ≥ 0.3.0 | Segmentación semántica de texto |

### LLMs soportados

| Proveedor | Modelo por defecto | Notas |
|-----------|------------------|-------|
| **Groq** | `llama-3.1-8b-instant` | Recomendado (gratis, rápido) |
| **OpenAI** | `gpt-4o-mini` | Requiere API key de pago |
| **Ollama** | `llama3.2` | Ejecución local, sin coste |
| **Flowise** | Configurable en el Agentflow | Agentes orquestados visualmente |

### Frontend

| Tecnología | Versión | Rol |
|-----------|---------|-----|
| **Streamlit** | ≥ 1.35.0 | Interfaz visual web |
| **Plotly** | ≥ 5.20.0 | Gráficos interactivos |
| **Pandas** | ≥ 1.4.0 | Manipulación de datos tabulares |

---

## 4. Estructura del Proyecto

```
FLOWISE/
│
├── main.py                    # Punto de entrada — FastAPI app + lifespan
├── streamlit_app.py           # Interfaz visual (3 pantallas)
├── requirements.txt           # Dependencias Python
├── .env                       # Variables de entorno (no commitear)
├── .env.example               # Plantilla de variables de entorno
│
├── app/
│   ├── __init__.py
│   └── config.py              # Configuración centralizada via pydantic-settings
│
├── embeddings/
│   ├── __init__.py
│   └── embedder.py            # Singleton embedder (multilingual-e5-small)
│
├── vectorstore/
│   ├── __init__.py
│   └── chroma_store.py        # Wrapper singleton sobre ChromaDB
│
├── services/
│   ├── __init__.py
│   ├── pdf_service.py         # Pipeline de procesamiento de PDFs
│   └── agent_service.py       # Pipeline de agentes Python + texto sugerido
│
├── prompts/
│   ├── __init__.py
│   └── agent_prompts.py       # Prompts de los 6 agentes + texto sugerido
│
├── flowise/
│   ├── __init__.py
│   └── client.py              # Cliente HTTP para la API de Flowise
│
├── routes/
│   ├── __init__.py
│   ├── upload.py              # POST /api/v1/upload-pdf
│   ├── query.py               # POST /api/v1/query
│   └── admin.py               # GET/DELETE /api/v1/health|collection|chunks
│
├── chroma_db/                 # Base de datos vectorial persistente (auto-generado)
│   └── *.bin / chroma.sqlite3
│
└── flowise_patch/
    └── End/
        └── End.js             # Nodo End personalizado para Flowise Agentflow
```

---

## 5. Módulos y Componentes

### 5.1 Configuración Central

**Archivo:** [app/config.py](app/config.py)

Gestiona toda la configuración del sistema a través de variables de entorno, usando `pydantic-settings` para validación automática de tipos.

```python
class Settings(BaseSettings):
    # Servidor
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True

    # LLM — modo automático: Groq → OpenAI → Ollama
    LLM_PROVIDER: str = "auto"
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # Flowise
    FLOWISE_URL: str = "http://localhost:3000"
    FLOWISE_CHATFLOW_ID: str = ""
    USE_FLOWISE: bool = False
    FLOWISE_MAX_CONTEXT_CHARS: int = 1500

    # Embeddings
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION: str = "academic_thesis"

    # RAG y Chunking
    TOP_K: int = 5
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
```

El objeto `settings` es un singleton importado por todos los módulos.

---

### 5.2 Módulo de Embeddings

**Archivo:** [embeddings/embedder.py](embeddings/embedder.py)

Implementa el patrón **Singleton** para cargar el modelo de embeddings una sola vez durante el arranque del servidor.

**Modelo utilizado:** `intfloat/multilingual-e5-small`

| Característica | Valor |
|---------------|-------|
| Idiomas | Multilingüe (>100 idiomas, incluyendo español) |
| Dimensión de vector | 384 |
| Normalización | L2 (cosine similarity) |
| Prefijo para documentos | `"passage: "` |
| Prefijo para consultas | `"query: "` |

> **Nota importante:** El modelo multilingual-e5 requiere el uso de prefijos distintos para documentos (`passage:`) y consultas (`query:`). No respetar esta convención degrada severamente la calidad del retrieval semántico.

**Métodos públicos:**

| Método | Entrada | Salida | Descripción |
|--------|---------|--------|-------------|
| `embed_documents(texts)` | `List[str]` | `List[List[float]]` | Genera embeddings para indexación |
| `embed_query(query)` | `str` | `List[float]` | Genera embedding para búsqueda |
| `dimension` (property) | — | `int` | Dimensión del vector (384) |

---

### 5.3 Vector Store (ChromaDB)

**Archivo:** [vectorstore/chroma_store.py](vectorstore/chroma_store.py)

Abstracción Singleton sobre ChromaDB con persistencia en disco. Usa distancia coseno para similitud semántica.

**Inicialización en lifespan:**
```python
chroma_store.initialize()
# → Crea o abre la colección "academic_thesis"
# → Configura métrica coseno (hnsw:space = cosine)
```

**Métodos principales:**

| Método | Descripción |
|--------|-------------|
| `initialize()` | Conecta con ChromaDB persistente al arrancar |
| `add_documents(texts, metadatas, ids)` | Genera embeddings y almacena chunks |
| `query(query_text, top_k, where)` | Búsqueda semántica, retorna lista con `text`, `metadata`, `score` |
| `format_context(results)` | Convierte resultados a bloque de texto para prompt |
| `get_info()` | Estadísticas de la colección (total chunks, estado) |
| `reset()` | Elimina y recrea la colección (destructivo) |

**Metadatos almacenados por chunk:**

```json
{
  "source": "nombre_del_archivo.pdf",
  "page": 12,
  "chunk_id": "tesis_chunk_0042",
  "section_detected": "metodologia",
  "char_count": 743
}
```

---

### 5.4 Servicio de Procesamiento de PDFs

**Archivo:** [services/pdf_service.py](services/pdf_service.py)

Pipeline completo de ingesta documental en 4 etapas:

#### Etapa 1 — Extracción de texto
- Usa `pypdf.PdfReader` para extraer texto página a página
- Ignora páginas con menos de 30 caracteres (páginas en blanco o imágenes)

#### Etapa 2 — Limpieza de texto
```python
def clean_text(text: str) -> str:
    # Normaliza saltos de línea múltiples → máximo 2
    # Elimina espacios redundantes (preserva \n)
    # Elimina caracteres de control no imprimibles
```

#### Etapa 3 — Chunking semántico
- Usa `RecursiveCharacterTextSplitter` con separadores jerárquicos: `["\n\n", "\n", ". ", " ", ""]`
- Parámetros configurables: `CHUNK_SIZE=800`, `CHUNK_OVERLAP=150`
- Descarta chunks menores a 50 caracteres

#### Etapa 4 — Detección de secciones académicas
Detecta automáticamente la sección académica de cada chunk mediante **expresiones regulares** sobre los primeros 300 caracteres:

| Sección detectada | Patrón regex |
|------------------|-------------|
| `resumen` | `RESUMEN\|ABSTRACT` |
| `introduccion` | `INTRODUCCION\|INTRODUCTION\|INTRODUCCIÓN` |
| `planteamiento_problema` | `PLANTEAMIENTO DEL PROBLEMA` |
| `justificacion` | `JUSTIFICACIÓN` |
| `objetivos` | `OBJETIVOS?` |
| `hipotesis` | `HIPÓTESIS` |
| `marco_teorico` | `MARCO TEÓRICO` |
| `metodologia` | `METODOLOGÍA` |
| `resultados` | `RESULTADOS?` |
| `conclusiones` | `CONCLUSIONES?` |
| `referencias` | `BIBLIOGRAFÍA\|REFERENCIAS` |
| *(y 7 más)* | ... |
| `general` | *(fallback)* |

**Respuesta del pipeline:**
```json
{
  "filename": "tesis.pdf",
  "total_pages": 87,
  "pages_with_content": 84,
  "chunks": [...],
  "sections_found": {
    "introduccion": 12,
    "metodologia": 18,
    "resultados": 24,
    "general": 30
  }
}
```

---

### 5.5 Pipeline de Agentes Secuenciales

**Archivo:** [services/agent_service.py](services/agent_service.py)

Implementa el modo **Python puro** (sin Flowise). Ejecuta los 6 agentes de forma secuencial con **memoria acumulativa**: cada agente recibe el resumen de los agentes anteriores para construir su análisis sobre el trabajo previo.

#### Selección dinámica de LLM (`_get_texto_llm`)
Prioridad automática cuando `LLM_PROVIDER=auto`:
1. **Groq** — si `GROQ_API_KEY` está configurado
2. **OpenAI** — si `OPENAI_API_KEY` está configurado
3. **Ollama** — fallback local, siempre disponible

Groq es compatible con la API de OpenAI, por lo que se integra via `langchain-openai` apuntando a `https://api.groq.com/openai/v1`.

#### Tolerancia a errores de JSON (`_parse_json`)
El parser es tolerante: intenta 3 estrategias antes de fallar:
1. `json.loads()` directo
2. Extrae bloque ` ```json ... ``` ` (markdown code blocks)
3. Busca el JSON más externo con llaves balanceadas `{...}`

#### Función `generate_texto_sugerido`
Genera una versión mejorada de la sección analizada, compatible con ambos modos (Flowise y Python). Recibe:
- `original_context`: el texto RAG recuperado de ChromaDB
- `question`: la pregunta del evaluador
- `final_evaluation`: dict del Mentor Final (puntos fuertes, áreas mejora, recomendaciones)
- `investigador_findings`: dict del Investigador (debilidades, sugerencias)

---

### 5.6 Cliente Flowise

**Archivo:** [flowise/client.py](flowise/client.py)

Wrapper asíncrono sobre la API HTTP de Flowise (v3). Gestiona el formateo del payload para el Agentflow personalizado.

#### Protocolo de comunicación

El Agentflow en Flowise tiene un nodo `CustomFunction` llamado `initializeFlowState` que espera el campo `question` como un **JSON serializado como string** con la siguiente estructura:

```json
{
  "section_type": "rag_query",
  "section_text": "pregunta del evaluador",
  "retrieved_context": "fragmentos RAG de ChromaDB",
  "research_line": "",
  "match_type": "semantic_similarity"
}
```

#### Truncado de contexto (`_truncate_context`)
Los nodos LLM del Agentflow tienen `maxTokens` bajos (300–350 tokens). Para evitar JSON truncado en la respuesta, el contexto RAG se limita a `FLOWISE_MAX_CONTEXT_CHARS=1500` caracteres, preservando fragmentos completos (delimitados por `---`).

#### Respuesta SSE (`_parse_sse_response`)
Flowise puede devolver respuestas en formato **Server-Sent Events** incluso con `streaming: false`. El cliente maneja ambos formatos (JSON directo y SSE).

**Endpoint usado:**
```
POST {FLOWISE_URL}/api/v1/prediction/{FLOWISE_CHATFLOW_ID}
```

---

### 5.7 API REST

**Framework:** FastAPI · **Puerto:** 8000 · **Docs:** `http://localhost:8000/docs`

#### POST `/api/v1/upload-pdf`

**Archivo:** [routes/upload.py](routes/upload.py)

Recibe un PDF, lo procesa y almacena en ChromaDB.

| Parámetro | Tipo | Descripción |
|-----------|------|-------------|
| `file` | `UploadFile` (form-data) | Archivo PDF de la tesis |

**Validaciones:**
- Content-type: `application/pdf` o extensión `.pdf`
- Tamaño máximo: 50 MB
- Mínimo 100 bytes (PDF no vacío)

**Respuesta exitosa (200):**
```json
{
  "success": true,
  "filename": "tesis.pdf",
  "file_size_mb": 2.34,
  "total_pages": 87,
  "chunks_generated": 312,
  "chunks_stored": 312,
  "sections_found": {
    "metodologia": 45,
    "resultados": 60,
    "general": 207
  },
  "message": "✅ PDF procesado correctamente. 312 fragmentos almacenados en ChromaDB."
}
```

---

#### POST `/api/v1/query`

**Archivo:** [routes/query.py](routes/query.py)

Orquesta el pipeline RAG + agentes y retorna la evaluación completa.

**Body (JSON):**
```json
{
  "question": "Evalúa la formulación del problema de investigación",
  "top_k": 5,
  "session_id": "sesion-proyecto-2024"
}
```

| Campo | Tipo | Requerido | Descripción |
|-------|------|-----------|-------------|
| `question` | string (5–2000 chars) | ✅ | Pregunta o instrucción de evaluación |
| `top_k` | int (1–20) | ❌ (default: 5) | Chunks a recuperar de ChromaDB |
| `session_id` | string | ❌ | ID de sesión para historial en Flowise |

**Respuesta (200):**
```json
{
  "question": "Evalúa la formulación del problema de investigación",
  "mode": "flowise",
  "chunks_retrieved": 5,
  "elapsed_seconds": 12.4,
  "context_preview": "[Fragmento 1 | Página 8 | Sección: planteamiento_problema]...",
  "result": {
    "flowise_response": { "text": "{...JSON evaluación...}" },
    "texto_sugerido": "Versión mejorada de la sección...",
    "original_context": "Texto original recuperado..."
  }
}
```

---

#### GET `/api/v1/health`

Devuelve el estado de todos los componentes del sistema.

**Respuesta:**
```json
{
  "status": "healthy",
  "components": {
    "chromadb": { "status": "connected", "chunks_stored": 312, "collection": "academic_thesis" },
    "embeddings": { "model": "intfloat/multilingual-e5-small", "status": "ready" },
    "llm": { "provider": "auto", "model": "gpt-4o-mini" },
    "flowise": { "url": "http://localhost:3000", "reachable": true, "mode": "active" }
  },
  "execution_mode": "flowise"
}
```

---

#### GET `/api/v1/collection`

Estadísticas de la colección ChromaDB activa.

#### DELETE `/api/v1/collection?confirm=true`

Elimina todos los chunks. Requiere `?confirm=true` como protección contra borrados accidentales.

#### GET `/api/v1/chunks?limit=10&offset=0`

Lista una muestra de los chunks almacenados con su texto (preview de 200 chars) y metadatos.

---

### 5.8 Interfaz Visual (Streamlit)

**Archivo:** [streamlit_app.py](streamlit_app.py)  
**Puerto:** 8501 (por defecto de Streamlit)

La interfaz consta de **3 pantallas** accesibles desde la barra lateral:

#### Pantalla 1 — 📄 Cargar PDF
- File uploader con drag & drop
- Configuración de `chunk_size` y `overlap` via sliders
- Métricas post-procesamiento: páginas, chunks generados, tamaño
- Gráfico de barras horizontal con secciones académicas detectadas
- Zona peligrosa para reiniciar ChromaDB

#### Pantalla 2 — 🔬 Ver Embeddings
Explora la representación vectorial del documento:
- **Donut chart:** distribución de chunks por sección académica
- **Histograma:** distribución de tamaños de chunks (en caracteres)
- **Scatter plot:** tamaño por posición en el documento (coloreado por sección)
- **Bar chart:** chunks generados por página del PDF
- **Tabla interactiva:** filtro por sección y búsqueda de texto en chunks

#### Pantalla 3 — 💬 Consultar Agentes
- Selector de preguntas de ejemplo (5 predefinidas)
- Área de texto libre para preguntas personalizadas
- Configuración avanzada: `top_k`, `session_id`
- Visualización estructurada de la respuesta del Mentor Final:
  - Puntuación general + nivel de tesis
  - Resumen ejecutivo
  - Retroalimentación pedagógica
  - Puntos fuertes / Áreas de mejora (lado a lado)
  - Recomendaciones priorizadas (expandibles)
  - Siguiente paso recomendado
- Comparador lado a lado: **texto original analizado** vs **texto mejorado sugerido**
- Historial de consultas de la sesión

**Estado del sistema en sidebar:**
- Indicador de conexión con el backend FastAPI
- Métrica de chunks en ChromaDB
- Estado de Flowise (alcanzable / sin conexión)
- Modo de ejecución activo

---

## 6. Los 6 Agentes Especializados

**Archivo:** [prompts/agent_prompts.py](prompts/agent_prompts.py)

El pipeline de evaluación consta de 6 agentes LLM que trabajan **secuencialmente con memoria acumulativa**. Cada agente recibe las evaluaciones de todos los anteriores y responde en **JSON estructurado**.

```
Contexto RAG
    │
    ▼
[1] MENTOR INTAKE ──→ memory["mentor_intake"]
    │
    ▼
[2] INVESTIGADOR ──→ memory["investigador"]
    │
    ▼
[3] AUDITOR ──→ memory["auditor"]
    │
    ▼
[4] METODOLÓGICO ──→ memory["metodologico"]
    │
    ▼
[5] REDACTOR ──→ memory["redactor"]
    │
    ▼
[6] MENTOR FINAL ──→ RESPUESTA FINAL (síntesis pedagógica)
```

### Agente 1 — Mentor de Evaluación Inicial (Mentor Intake)

**Rol:** Evaluación inicial del fragmento recuperado.

**Tareas:**
1. Identificar el tema central y la sección académica presente
2. Evaluar si el contexto es suficiente para responder la pregunta
3. Identificar los 3 aspectos clave a evaluar
4. Señalar limitaciones del contexto recuperado

**Output JSON:**
```json
{
  "tema_identificado": "string",
  "seccion_relevante": "string",
  "pertinencia_pregunta": "alta|media|baja",
  "contexto_suficiente": true,
  "aspectos_clave": ["aspecto1", "aspecto2", "aspecto3"],
  "evaluacion_inicial": "string",
  "limitaciones_contexto": ["limitacion1"],
  "flags": []
}
```

---

### Agente 2 — Investigador

**Rol:** Análisis de la calidad investigativa.

**Tareas:**
1. Analizar la solidez argumentativa
2. Evaluar respaldo teórico y bibliográfico
3. Identificar fortalezas y debilidades investigativas
4. Sugerir 2-3 mejoras específicas
5. Asignar puntuación de 0 a 10

**Output JSON:**
```json
{
  "fortalezas": ["string"],
  "debilidades": ["string"],
  "respaldo_teorico": "adecuado|parcial|insuficiente",
  "relevancia_cientifica": "alta|media|baja",
  "sugerencias": ["string"],
  "puntuacion": 7.5,
  "comentario": "string"
}
```

---

### Agente 3 — Auditor

**Rol:** Rigor académico y coherencia científica.

**Tareas:**
1. Verificar coherencia interna del argumento
2. Detectar inconsistencias o afirmaciones sin soporte
3. Evaluar terminología académica
4. Identificar brechas o vacíos argumentales
5. Señalar problemas críticos

**Output JSON:**
```json
{
  "nivel_rigor": "alto|medio|bajo",
  "coherencia_interna": "alta|media|baja",
  "inconsistencias": ["string"],
  "terminologia": "correcta|parcialmente_correcta|incorrecta",
  "brechas_detectadas": ["string"],
  "problemas_criticos": ["string"],
  "puntuacion_rigor": 7.0,
  "recomendaciones": ["string"]
}
```

---

### Agente 4 — Metodológico

**Rol:** Análisis del marco y diseño metodológico.

**Tareas:**
1. Identificar el enfoque (cualitativo/cuantitativo/mixto)
2. Evaluar adecuación del diseño al problema
3. Analizar instrumentos de recolección
4. Identificar limitaciones metodológicas
5. Sugerir ajustes metodológicos

**Output JSON:**
```json
{
  "enfoque": "cualitativo|cuantitativo|mixto|no_especificado",
  "tipo_investigacion": "descriptiva|explicativa|correlacional|experimental|exploratoria|mixta",
  "diseno": "string",
  "adecuacion_metodologica": "alta|media|baja",
  "instrumentos_identificados": ["string"],
  "limitaciones_metodologicas": ["string"],
  "sugerencias_metodologicas": ["string"],
  "puntuacion_metodologia": 7.0,
  "comentario": "string"
}
```

---

### Agente 5 — Redactor

**Rol:** Mejora de escritura académica.

**Tareas:**
1. Seleccionar el fragmento más relevante del contexto
2. Reescribir con mayor claridad y estilo académico
3. Mantener el significado original (solo mejora la forma)
4. Listar cambios específicos
5. Proveer sugerencias generales de escritura

**Output JSON:**
```json
{
  "fragmento_original": "string",
  "fragmento_mejorado": "string",
  "cambios_realizados": ["string"],
  "nivel_escritura_original": "alto|medio|bajo",
  "sugerencias_generales": ["string"],
  "comentario": "string"
}
```

---

### Agente 6 — Mentor Final

**Rol:** Síntesis pedagógica y feedback constructivo al estudiante.

**Tareas:**
1. Sintetizar hallazgos de todos los agentes anteriores
2. Identificar los 3 puntos fuertes principales
3. Listar las 3 áreas de mejora más urgentes
4. Generar recomendaciones priorizadas (máx. 5)
5. Calcular puntuación general (promedio ponderado)
6. Redactar mensaje motivador y pedagógico
7. Indicar el siguiente paso concreto

**Output JSON:**
```json
{
  "resumen_ejecutivo": "string",
  "puntos_fuertes": ["string", "string", "string"],
  "areas_mejora": ["string", "string", "string"],
  "recomendaciones_priorizadas": [
    {"prioridad": 1, "recomendacion": "string", "justificacion": "string"},
    {"prioridad": 2, "recomendacion": "string", "justificacion": "string"},
    {"prioridad": 3, "recomendacion": "string", "justificacion": "string"}
  ],
  "puntuacion_general": 7.2,
  "nivel_tesis": "excelente|buena|aceptable|necesita_mejoras|insuficiente",
  "mensaje_pedagogico": "string",
  "siguiente_paso": "string"
}
```

---

## 7. Flujo de Datos Completo

### Flujo de Ingesta (POST /upload-pdf)

```
Usuario sube PDF
      │
      ▼
FastAPI valida (tipo, tamaño, contenido)
      │
      ▼
pdf_service.process_pdf()
      ├─ extract_pages()         → texto por página (pypdf)
      ├─ clean_text()            → normalización
      ├─ build_chunks()          → segmentos de 800 chars, overlap 150
      └─ detect_section()        → clasificación académica por regex
      │
      ▼
embedder.embed_documents()       → vectores 384D (multilingual-e5-small)
      │
      ▼
chroma_store.add_documents()     → persiste en ./chroma_db
      │
      ▼
Respuesta: {chunks_stored, sections_found, ...}
```

### Flujo de Consulta (POST /query)

```
Usuario envía pregunta
      │
      ▼
FastAPI valida body
      │
      ▼
chroma_store.query(question, top_k=5)
      ├─ embedder.embed_query()   → vector de consulta ("query: ...")
      └─ ChromaDB similarity     → top-K chunks más similares
      │
      ▼
chroma_store.format_context()    → texto con metadatos para el prompt
      │
      ├─[USE_FLOWISE=true]──────────────────────────────────────────┐
      │                                                             │
      │                                              flowise_client.call_chatflow()
      │                                                    │
      │                                              Flowise Agentflow
      │                                                    │
      │                                              6 nodos LLM en Flowise
      │                                              (Groq configurado en Flowise)
      │                                                    │
      │                                              JSON evaluación final
      │                                                    │
      └─[USE_FLOWISE=false]─────────────────────────────────────────┘
                                                         │
                                              agent_service.run_sequential_pipeline()
                                                    ├─ Agente 1: Mentor Intake
                                                    ├─ Agente 2: Investigador
                                                    ├─ Agente 3: Auditor
                                                    ├─ Agente 4: Metodológico
                                                    ├─ Agente 5: Redactor
                                                    └─ Agente 6: Mentor Final
                                                         │
      ┌──────────────────────────────────────────────────┘
      │
      ▼
generate_texto_sugerido()        → versión mejorada de la sección (Groq/OpenAI/Ollama)
      │
      ▼
QueryResponse: {question, mode, chunks_retrieved, elapsed_seconds, result}
```

---

## 8. API Reference

### Resumen de endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/` | Raíz — mapa de endpoints |
| `POST` | `/api/v1/upload-pdf` | Procesar e indexar un PDF |
| `POST` | `/api/v1/query` | Consulta RAG + evaluación multiagente |
| `GET` | `/api/v1/health` | Estado de todos los componentes |
| `GET` | `/api/v1/collection` | Info de la colección ChromaDB |
| `DELETE` | `/api/v1/collection?confirm=true` | Reiniciar colección (destructivo) |
| `GET` | `/api/v1/chunks?limit=10&offset=0` | Listar chunks almacenados |

### Docs interactivas
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

### Códigos de error

| Código | Descripción |
|--------|-------------|
| `400` | Archivo no válido / operación sin confirmación |
| `404` | No hay PDF cargado / sin fragmentos relevantes |
| `413` | PDF demasiado grande (>50 MB) |
| `422` | PDF sin texto extraíble (posiblemente escaneado) |
| `500` | Error interno (extracción, embeddings, agentes) |
| `502` | Flowise no disponible o chatflow ID incorrecto |

---

## 9. Variables de Entorno

**Archivo:** `.env` (basado en `.env.example`)

```bash
# ── SERVIDOR ──────────────────────────────────────────────────────────
HOST=0.0.0.0
PORT=8000
DEBUG=true

# ── FLOWISE ───────────────────────────────────────────────────────────
USE_FLOWISE=true                                   # true: Flowise | false: Python puro
FLOWISE_URL=http://localhost:3000
FLOWISE_CHATFLOW_ID=8ef396a1-4ae8-45b7-93cb-5b449e928854
FLOWISE_API_KEY=                                   # opcional si hay autenticación en Flowise

# ── LLM (para texto sugerido y modo Python puro) ─────────────────────
LLM_PROVIDER=auto                                  # auto | groq | openai | ollama
GROQ_API_KEY=gsk_...                               # gratis en console.groq.com
GROQ_MODEL=llama-3.1-8b-instant
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini
# OLLAMA_BASE_URL=http://localhost:11434
# OLLAMA_MODEL=llama3.2

# ── EMBEDDINGS ────────────────────────────────────────────────────────
EMBEDDING_MODEL=intfloat/multilingual-e5-small

# ── CHROMADB ─────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION=academic_thesis

# ── RAG ──────────────────────────────────────────────────────────────
TOP_K=5                                            # chunks por consulta

# ── CHUNKING ─────────────────────────────────────────────────────────
CHUNK_SIZE=800                                     # caracteres por chunk
CHUNK_OVERLAP=150                                  # solapamiento entre chunks
```

---

## 10. Modos de Ejecución

### Modo A: Flowise Agentflow (`USE_FLOWISE=true`)

Los agentes corren dentro de Flowise como un flujo visual de nodos LLM:
- Requiere Flowise corriendo en `localhost:3000`
- El `FLOWISE_CHATFLOW_ID` apunta al Agentflow con los 6 nodos
- Los modelos LLM (Groq) se configuran directamente en Flowise, no en `.env`
- El contexto RAG se trunca a 1500 chars para respetar `maxTokens` de los nodos
- El texto sugerido sí usa el LLM configurado en `.env` (Groq/OpenAI/Ollama)

**Ventajas:** Visual, editable sin código, trazabilidad de nodos  
**Desventajas:** Requiere Flowise corriendo, más latencia por HTTP

### Modo B: Agentes Python puro (`USE_FLOWISE=false`)

Los 6 agentes se ejecutan directamente en Python via LangChain:
- No requiere Flowise
- Usa el LLM configurado en `.env` (Groq → OpenAI → Ollama)
- Cada agente hace una llamada independiente al LLM
- Memoria completamente controlada en Python

**Ventajas:** Sin dependencias externas, ideal para testing y debugging  
**Desventajas:** 6 llamadas LLM por query (mayor latencia y coste si OpenAI)

---

## 11. Instrucciones de Instalación y Ejecución

### Requisitos previos
- Python 3.10+ (el proyecto usa 3.14)
- pip o pipenv
- (Opcional) Flowise instalado y corriendo
- (Opcional) Ollama instalado localmente

### Instalación

```bash
# 1. Clonar el repositorio
cd FLOWISE

# 2. Crear entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
copy .env.example .env
# Editar .env con tus claves de API
```

### Ejecución

```bash
# Terminal 1 — Backend FastAPI
python main.py
# → http://localhost:8000
# → Docs: http://localhost:8000/docs

# Terminal 2 — Frontend Streamlit
python -m streamlit run streamlit_app.py
# → http://localhost:8501
```

### Uso básico

```bash
# 1. Subir una tesis
curl -X POST http://localhost:8000/api/v1/upload-pdf \
     -F "file=@mi_tesis.pdf"

# 2. Consultar
curl -X POST http://localhost:8000/api/v1/query \
     -H "Content-Type: application/json" \
     -d '{"question": "Evalúa el marco metodológico", "top_k": 5}'

# 3. Verificar estado
curl http://localhost:8000/api/v1/health
```

---

## 12. Diagrama de Flujo de Agentes

### Flujo de memoria acumulativa

```
┌─────────────────────────────────────────────────────────────────┐
│  CONTEXTO RAG (top-K chunks de ChromaDB)                        │
│  PREGUNTA DEL EVALUADOR                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │   AGENTE 1            │
                 │   Mentor Intake       │
                 │   ─────────────────── │
                 │   IN: contexto,       │
                 │       pregunta        │
                 │   OUT: tema, sección, │
                 │   aspectos_clave,     │
                 │   evaluacion_inicial  │
                 └───────────┬───────────┘
                             │ memory["mentor_intake"]
                             ▼
                 ┌───────────────────────┐
                 │   AGENTE 2            │
                 │   Investigador        │
                 │   ─────────────────── │
                 │   IN: + memory[1]     │
                 │   OUT: fortalezas,    │
                 │   debilidades,        │
                 │   puntuacion (0-10)   │
                 └───────────┬───────────┘
                             │ memory["investigador"]
                             ▼
                 ┌───────────────────────┐
                 │   AGENTE 3            │
                 │   Auditor             │
                 │   ─────────────────── │
                 │   IN: + memory[1,2]   │
                 │   OUT: rigor,         │
                 │   coherencia,         │
                 │   problemas_criticos  │
                 └───────────┬───────────┘
                             │ memory["auditor"]
                             ▼
                 ┌───────────────────────┐
                 │   AGENTE 4            │
                 │   Metodológico        │
                 │   ─────────────────── │
                 │   IN: + memory[1,2,3] │
                 │   OUT: enfoque,       │
                 │   tipo_investigacion, │
                 │   limitaciones        │
                 └───────────┬───────────┘
                             │ memory["metodologico"]
                             ▼
                 ┌───────────────────────┐
                 │   AGENTE 5            │
                 │   Redactor            │
                 │   ─────────────────── │
                 │   IN: + memory[1,3,4] │
                 │   OUT: fragmento_     │
                 │   mejorado,           │
                 │   cambios_realizados  │
                 └───────────┬───────────┘
                             │ memory["redactor"]
                             ▼
                 ┌───────────────────────┐
                 │   AGENTE 6            │
                 │   Mentor Final        │
                 │   ─────────────────── │
                 │   IN: memory completa │
                 │   OUT: resumen_exec,  │
                 │   recomendaciones_    │
                 │   priorizadas,        │
                 │   puntuacion_general, │
                 │   mensaje_pedagogico, │
                 │   siguiente_paso      │
                 └───────────┬───────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   POST-PIPELINE              │
              │   Generador Texto Sugerido   │
              │   ──────────────────────     │
              │   IN: contexto original,     │
              │       eval. Mentor Final,    │
              │       hallazgos Investigador │
              │   OUT: texto_sugerido        │
              │   (para reemplazar sección)  │
              └──────────────────────────────┘
```

---

*Documentación generada automáticamente el 23 de mayo de 2026.*  
*Sistema: POC Evaluador de Tesis RAG Multiagente v1.0.0*
