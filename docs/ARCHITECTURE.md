# Legal-NER System Architecture

> **Documento di Architettura Tecnica**: Descrizione dettagliata dell'architettura, componenti e flussi di dati del sistema Legal-NER.

## Indice

1. [Overview Architetturale](#overview-architetturale)
2. [Architettura a Livelli](#architettura-a-livelli)
3. [Componenti Core](#componenti-core)
4. [Pipeline NER Multi-Stage](#pipeline-ner-multi-stage)
5. [Sistema di Active Learning](#sistema-di-active-learning)
6. [Data Flow](#data-flow)
7. [Modelli di Machine Learning](#modelli-di-machine-learning)
8. [Storage e Persistenza](#storage-e-persistenza)
9. [Sistema di Configurazione](#sistema-di-configurazione)
10. [API Layer](#api-layer)
11. [Scalabilità e Performance](#scalabilita-e-performance)
12. [Sicurezza](#sicurezza)

---

## Overview Architetturale

### Principi Architetturali

Il sistema Legal-NER è progettato seguendo questi principi:

1. **Separation of Concerns**: Ogni componente ha una responsabilità specifica
2. **Configuration-Driven**: Comportamento controllato da file YAML esterni
3. **Modularity**: Componenti sostituibili e testabili indipendentemente
4. **Scalability**: Design che supporta crescita di dati e utenti
5. **Observability**: Logging strutturato JSON per monitoring
6. **Type Safety**: Type hints Python per maggiore sicurezza

### Stack Tecnologico Completo

```
┌─────────────────────────────────────────────────────────┐
│                    Technology Stack                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Application Layer                                       │
│  ├─ FastAPI 0.104.1        (Async REST API)            │
│  ├─ Flask 3.0.0            (UI Server)                  │
│  ├─ Pydantic 2.5.0         (Data Validation)           │
│  └─ Structlog 23.2.0       (Structured Logging)        │
│                                                          │
│  ML/NER Layer                                            │
│  ├─ Transformers 4.35.0    (HuggingFace Models)        │
│  ├─ PyTorch 2.1.0          (Deep Learning)             │
│  ├─ Sentence-Transformers  (Embeddings)                │
│  └─ scikit-learn 1.3.2     (ML Utilities)              │
│                                                          │
│  Data Layer                                              │
│  ├─ PostgreSQL 15          (Relational Database)        │
│  ├─ SQLAlchemy 2.0.23      (ORM)                       │
│  ├─ Alembic 1.12.1         (Migrations)                │
│  └─ MinIO                  (Object Storage)             │
│                                                          │
│  Infrastructure                                          │
│  ├─ Docker + Docker Compose                            │
│  ├─ Nginx (Reverse Proxy)                              │
│  └─ Python 3.13+                                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Architettura a Livelli

### Layered Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      Presentation Layer                         │
│  ┌──────────────────────────┐  ┌──────────────────────────┐   │
│  │      Flask UI            │  │    FastAPI REST API      │   │
│  │  • Web Interface         │  │  • RESTful Endpoints     │   │
│  │  • Jinja2 Templates      │  │  • OpenAPI/Swagger       │   │
│  │  • Bootstrap 5           │  │  • Request Validation    │   │
│  └──────────────────────────┘  └──────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                       Business Logic Layer                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              Service Layer (app/services/)                │ │
│  │  ┌────────────────────────────────────────────────────┐  │ │
│  │  │         Specialized NER Pipeline                    │  │ │
│  │  │  • EntityDetector                                   │  │ │
│  │  │  • LegalClassifier                                  │  │ │
│  │  │  • NormativeParser                                  │  │ │
│  │  │  • ReferenceResolver                                │  │ │
│  │  │  • StructureBuilder                                 │  │ │
│  │  └────────────────────────────────────────────────────┘  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │       Active Learning Layer (app/feedback/)             │ │
│  │  • ActiveLearningManager                                 │ │
│  │  • DatasetBuilder                                        │ │
│  │  • Uncertainty Sampling                                  │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         Core Utilities (app/core/)                       │ │
│  │  • Configuration Loaders                                 │ │
│  │  • Label Mapping System                                  │ │
│  │  • Model Manager                                         │ │
│  │  • Logging                                               │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                      Data Access Layer                          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │          Database Layer (app/database/)                  │ │
│  │  • SQLAlchemy ORM Models                                 │ │
│  │  • CRUD Operations                                       │ │
│  │  • Database Session Management                           │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────┐
│                      Infrastructure Layer                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────┐ │
│  │   PostgreSQL     │  │      MinIO       │  │   Config    │ │
│  │   (Database)     │  │  (Object Store)  │  │   (YAML)    │ │
│  └──────────────────┘  └──────────────────┘  └─────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

---

## Componenti Core

### 1. FastAPI Application (`app/main.py`)

**Responsabilità**:
- Entry point dell'applicazione backend
- Gestione routing API
- Middleware configuration
- Lifecycle management (startup/shutdown)

**Caratteristiche**:
```python
app = FastAPI(
    title="Legal-NER-API",
    version="1.0.0",
    description="API for NER in Italian legal texts"
)

# Middlewares
- CORSMiddleware: Cross-origin requests
- LoggingMiddleware: Request/response logging
- Custom JSON encoders: numpy/torch serialization

# Routers registrati:
- /api/v1/predict          # NER prediction
- /api/v1/feedback         # User feedback
- /api/v1/active-learning  # Active learning
- /api/v1/documents        # Document management
- /api/v1/annotations      # Annotations
- /api/v1/process          # Document processing
- /api/v1/export           # Data export
- /api/v1/models           # Model management
- /api/v1/labels           # Label management
- /api/v1/admin            # Admin operations
```

### 2. Specialized NER Pipeline (`app/services/specialized_pipeline.py`)

**Architettura Pipeline**:

```
Input: str (testo legale)
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ Stage 1: EntityDetector                             │
│                                                      │
│ Models:                                              │
│  • Primary: Italian-Legal-BERT                      │
│  • Fallback: WikiNEural-Multilingual                │
│                                                      │
│ Processo:                                            │
│  1. Tokenization con offset mapping                 │
│  2. Model inference (BIO tagging)                   │
│  3. Rule-based fallback se necessario               │
│  4. Espansione confini riferimenti                  │
│  5. Filtro sovrapposizioni                          │
│                                                      │
│ Output: List[TextSpan]                              │
│  • text: str                                         │
│  • start_char: int (nativo Python)                  │
│  • end_char: int (nativo Python)                    │
│  • initial_confidence: float (nativo Python)        │
│  • context_window: str                              │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: LegalClassifier                            │
│                                                      │
│ Strategia Multi-Approccio:                          │
│                                                      │
│ A) Fine-tuned Model (se disponibile):               │
│    • Token-level classification                     │
│    • BIO encoding                                   │
│    • Confidence threshold: 0.7                      │
│                                                      │
│ B) Rule-based Classification:                       │
│    • Pattern matching prioritizzato                 │
│    • 500+ abbreviazioni mappate                     │
│    • Confidence per pattern: 0.50 - 0.99           │
│                                                      │
│ C) Semantic Validation:                             │
│    • Italian-Legal-BERT embeddings                  │
│    • Cosine similarity con prototipi                │
│    • 300+ prototipi semantici                       │
│                                                      │
│ Logica Decisionale:                                 │
│  1. Fine-tuned model se conf >= 0.7                │
│  2. Rule-based classification                       │
│  3. Semantic validation se:                         │
│     - Rule confidence < threshold                   │
│     - enable_semantic_validation_always = true      │
│  4. Gestione discrepanze:                           │
│     - Rule >= 0.90 → vince sempre                  │
│     - Semantic < threshold → UNKNOWN                │
│     - Semantic >> Rule (+0.15) → semantic vince    │
│                                                      │
│ Output: LegalClassification                         │
│  • span: TextSpan                                   │
│  • act_type: str (es. "decreto_legislativo")       │
│  • confidence: float                                │
│  • semantic_embedding: Optional[np.ndarray]         │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: NormativeParser                            │
│                                                      │
│ Pattern Extraction:                                  │
│  • act_number: regex pattern                        │
│  • date: 3 pattern con priorità                     │
│  • article: articolo/art. + numero                  │
│  • comma: comma/co. + numero                        │
│  • letter: lettera/lett. + lettera                  │
│  • paragraph: paragrafo/par. + numero               │
│  • annex: allegato + lettera                        │
│  • version: modifiche/versioni                      │
│                                                      │
│ Special Handling:                                    │
│  • Direttive UE: formato anno/numero                │
│  • Numeri bis/ter/quater                            │
│  • Date multiple formati                            │
│                                                      │
│ Output: ParsedNormative                             │
│  • text: str                                         │
│  • act_type: str                                    │
│  • act_number: Optional[str]                        │
│  • date: Optional[str]                              │
│  • article: Optional[str]                           │
│  • comma: Optional[str]                             │
│  • letter: Optional[str]                            │
│  • version: Optional[str]                           │
│  • is_complete_reference: bool                      │
│  • confidence: float                                │
│  • start_char, end_char: int                        │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4: ReferenceResolver                          │
│                                                      │
│ Funzionalità (Current):                             │
│  • Direct resolution (pass-through)                 │
│  • Resolution metadata                              │
│                                                      │
│ Funzionalità (Future):                              │
│  • Implicit reference resolution                    │
│  • Reference chain following                        │
│  • Context-based completion                         │
│  • Cross-document references                        │
│                                                      │
│ Output: ResolvedNormative                           │
│  • Inherits from ParsedNormative                    │
│  • resolution_method: str                           │
│  • resolution_confidence: float                     │
└─────────────────────────────────────────────────────┘
   │
   ▼
┌─────────────────────────────────────────────────────┐
│ Stage 5: StructureBuilder                           │
│                                                      │
│ Transformations:                                     │
│  1. Convert act_type → standardized label           │
│     (via label_mapping system)                      │
│                                                      │
│  2. Apply filters:                                  │
│     • filter_institutions: remove institutions      │
│     • filter_null_values: remove null fields        │
│                                                      │
│  3. Add metadata:                                   │
│     • parsing_timestamp                             │
│     • model_version                                 │
│     • config_hash                                   │
│                                                      │
│  4. Quality checks:                                 │
│     • confidence >= threshold                       │
│     • completeness validation                       │
│     • reference validation                          │
│                                                      │
│ Output: Dict[str, Any]                              │
│  {                                                   │
│    "source_type": "CODICE_CIVILE",  # Label std    │
│    "text": "art. 2043 c.c.",                       │
│    "confidence": 0.99,                              │
│    "start_char": 12,                                │
│    "end_char": 26,                                  │
│    "act_type": "codice_civile",                    │
│    "article": "2043",                               │
│    "date": null,                                    │
│    ...                                              │
│  }                                                   │
└─────────────────────────────────────────────────────┘
   │
   ▼
Output: List[Dict[str, Any]] (entità strutturate)
```

**Conversione Tipi Nativi**:

La pipeline garantisce che TUTTI i valori numerici siano convertiti a tipi Python nativi prima dell'output:

```python
# Conversioni automatiche:
torch.Tensor → int/float (via .item())
np.integer → int (via .item())
np.floating → float (via .item())

# Punti di conversione:
1. EntityDetector._extract_entities_with_offsets()
   - start_char, end_char: Tensor → int
   - confidence: float32 → float

2. EntityDetector._expand_reference_boundaries()
   - start_char, end_char: verificati e convertiti

3. Output finale pipeline
   - Tutti i dict values verificati prima del return
```

### 3. Active Learning System (`app/feedback/active_learning.py`)

**Architettura Active Learning**:

```
┌─────────────────────────────────────────────────────────────┐
│                 ActiveLearningManager                        │
│                                                               │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  1. Uncertainty Sampling                                │ │
│  │                                                          │ │
│  │  Process:                                                │ │
│  │  a) Query database per documenti senza annotazioni      │ │
│  │  b) Run NER pipeline su ciascun documento              │ │
│  │  c) Calcola uncertainty score:                          │ │
│  │                                                          │ │
│  │     uncertainty = 1 - avg(confidence_entities)          │ │
│  │                                                          │ │
│  │  d) Seleziona top-N documenti per uncertainty          │ │
│  │  e) Filtra documenti già in task pendenti              │ │
│  │                                                          │ │
│  │  Strategies disponibili:                                │ │
│  │  - uncertainty: max incertezza (default)                │ │
│  │  - random: selezione casuale                            │ │
│  │  - diversity: max diversità (TODO)                      │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                   │
│                           ▼                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  2. Task Creation                                       │ │
│  │                                                          │ │
│  │  Per ogni documento selezionato:                        │ │
│  │  a) Crea AnnotationTask:                                │ │
│  │     - document_id                                       │ │
│  │     - status: "pending"                                 │ │
│  │     - priority: uncertainty_score                       │ │
│  │     - created_at: now()                                 │ │
│  │                                                          │ │
│  │  b) Salva entità pre-estratte:                          │ │
│  │     - Applica label_mapping per consistenza            │ │
│  │     - Converte tipi a Python nativi                    │ │
│  │     - Link a annotation_task                           │ │
│  │                                                          │ │
│  │  c) Transaction commit                                  │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                   │
│                           ▼                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  3. Feedback Collection (via UI/API)                   │ │
│  │                                                          │ │
│  │  User actions:                                          │ │
│  │  - Validate entity: is_correct = true/false            │ │
│  │  - Correct label: corrected_label = new_label         │ │
│  │  - Add feedback text (optional)                         │ │
│  │  - Add new entities (manual annotation)                │ │
│  │  - Delete wrong entities                                │ │
│  │                                                          │ │
│  │  Storage:                                                │ │
│  │  - Annotation record per ogni feedback                  │ │
│  │  - Linked to entity_id e user_id                       │ │
│  │  - Timestamp per tracking                               │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                   │
│                           ▼                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  4. Dataset Building                                    │ │
│  │                                                          │ │
│  │  DatasetBuilder process:                                │ │
│  │  a) Query annotated documents                           │ │
│  │  b) Extract entities con annotations                    │ │
│  │  c) Convert to IOB format:                              │ │
│  │                                                          │ │
│  │     Input:  "art. 2043 c.c."                           │ │
│  │     Tokens: ["art.", "2043", "c.c."]                   │ │
│  │     Tags:   ["B-CODICE_CIVILE",                        │ │
│  │              "I-CODICE_CIVILE",                        │ │
│  │              "I-CODICE_CIVILE"]                        │ │
│  │                                                          │ │
│  │  d) Split train/validation/test (80/10/10)            │ │
│  │  e) Save to MinIO                                       │ │
│  │  f) Return dataset metadata                             │ │
│  └────────────────────────────────────────────────────────┘ │
│                           │                                   │
│                           ▼                                   │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  5. Model Training                                      │ │
│  │                                                          │ │
│  │  Training pipeline:                                      │ │
│  │  a) Load dataset da MinIO                               │ │
│  │  b) Load base model (Italian-Legal-BERT)               │ │
│  │  c) Fine-tune with HuggingFace Trainer:                │ │
│  │     - Task: Token Classification                        │ │
│  │     - Loss: CrossEntropyLoss                           │ │
│  │     - Optimizer: AdamW                                  │ │
│  │     - Scheduler: Linear warmup + decay                 │ │
│  │     - Epochs: configurabile (default 3)                │ │
│  │     - Batch size: configurabile (default 16)           │ │
│  │                                                          │ │
│  │  d) Evaluate on validation set:                         │ │
│  │     - Precision, Recall, F1 per label                  │ │
│  │     - Overall accuracy                                  │ │
│  │     - Confusion matrix                                  │ │
│  │                                                          │ │
│  │  e) Save model to MinIO:                               │ │
│  │     - Model weights                                     │ │
│  │     - Tokenizer                                         │ │
│  │     - Config                                            │ │
│  │     - Label mapping                                     │ │
│  │                                                          │ │
│  │  f) Register in database:                               │ │
│  │     - MLModel record                                    │ │
│  │     - Metrics JSON                                      │ │
│  │     - Version info                                      │ │
│  │                                                          │ │
│  │  g) Optional: activate if metrics > threshold          │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Uncertainty Calculation**:

```python
def calculate_uncertainty(entities: List[Dict]) -> float:
    """
    Calcola uncertainty di un documento basata su confidence entità.

    Formula:
    uncertainty = 1 - (sum(confidence) / count(entities))

    Esempi:
    - 3 entità con conf [0.99, 0.98, 0.97] → uncertainty = 0.02
    - 3 entità con conf [0.60, 0.55, 0.50] → uncertainty = 0.45
    - 0 entità → uncertainty = 1.0 (max incertezza)

    Interpretazione:
    - uncertainty > 0.7: Alta priorità per annotazione
    - 0.5 < uncertainty < 0.7: Media priorità
    - uncertainty < 0.5: Bassa priorità
    """
    if not entities:
        return 1.0

    confidences = [e.get("confidence", 0.0) for e in entities]
    avg_confidence = sum(confidences) / len(confidences)

    return 1.0 - avg_confidence
```

### 4. Label Mapping System (`app/core/label_mapping.py`)

**Architettura Label Mapping**:

```
┌─────────────────────────────────────────────────────────┐
│              Label Mapping System                        │
│                                                           │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Configuration File (label_mapping.yaml)            │ │
│  │                                                      │ │
│  │  act_type_to_label:                                 │ │
│  │    decreto_legislativo: D.LGS                       │ │
│  │    codice_civile: CODICE_CIVILE                    │ │
│  │    ...                                               │ │
│  │                                                      │ │
│  │  label_categories:                                  │ │
│  │    Decreti: [D.LGS, D.L, ...]                      │ │
│  │    Codici: [CODICE_CIVILE, ...]                    │ │
│  │    ...                                               │ │
│  └────────────────────────────────────────────────────┘ │
│                           │                              │
│                           ▼                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Loader + Cache                                     │ │
│  │                                                      │ │
│  │  def _load_label_config():                          │ │
│  │    if _LABEL_CACHE is not None:                    │ │
│  │      return _LABEL_CACHE  # Fast path              │ │
│  │                                                      │ │
│  │    config = yaml.safe_load(LABEL_CONFIG_PATH)      │ │
│  │    _LABEL_CACHE = config                           │ │
│  │    return config                                    │ │
│  │                                                      │ │
│  │  Cache invalidation: manual via reload_label_config()│ │
│  └────────────────────────────────────────────────────┘ │
│                           │                              │
│                           ▼                              │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Public API                                         │ │
│  │                                                      │ │
│  │  # Conversion functions                             │ │
│  │  act_type_to_label(act_type: str) → str           │ │
│  │  label_to_act_type(label: str) → str              │ │
│  │                                                      │ │
│  │  # Query functions                                  │ │
│  │  get_all_labels() → List[str]                      │ │
│  │  get_label_categories() → Dict[str, List[str]]    │ │
│  │  get_label_category(label: str) → str             │ │
│  │  validate_label(label: str) → bool                │ │
│  │                                                      │ │
│  │  # Management functions                             │ │
│  │  update_label_mapping(act_type, label, category)   │ │
│  │  remove_label_mapping(act_type)                    │ │
│  │  reload_label_config()                             │ │
│  └────────────────────────────────────────────────────┘ │
│                                                           │
│  Usage Points:                                           │
│  ├─ admin.py: Reprocessing tasks                        │
│  ├─ process.py: Document processing                     │
│  ├─ active_learning.py: Task creation                   │
│  ├─ annotations.py: Manual annotations                  │
│  ├─ crud.py: Database operations                        │
│  └─ labels.py: API endpoints                            │
└─────────────────────────────────────────────────────────┘
```

**Fallback Strategy**:

```python
def act_type_to_label(act_type: str) -> str:
    """
    Conversione con fallback intelligente.

    1. Try exact match in mapping
    2. Try normalized match (lowercase, strip)
    3. Fallback: act_type.upper().replace(' ', '_')

    Esempi:
    - "decreto_legislativo" → "D.LGS" (exact match)
    - "Decreto Legislativo" → "D.LGS" (normalized)
    - "nuovo_tipo" → "NUOVO_TIPO" (fallback)
    """
```

---

## Data Flow

### Request Flow - Prediction

```
1. Client Request
   POST /api/v1/predict
   {
     "text": "Secondo l'art. 2043 c.c., chiunque cagiona danno..."
   }
   │
   ▼
2. FastAPI Router (predict.py)
   - Validate request (Pydantic)
   - Extract text
   - Generate request_id
   │
   ▼
3. Create Log File
   logs/requests/{request_id}.log
   │
   ▼
4. Get Active Pipeline
   model_manager.get_active_pipeline(db)
   - Query database for active model
   - Load fine-tuned model if available
   - Or use rule-based pipeline
   │
   ▼
5. Execute Pipeline
   pipeline.extract_legal_sources(text, log_file_path)
   │
   ├─▶ Stage 1: EntityDetector
   │   - BERT inference
   │   - Rule-based fallback
   │   - Boundary expansion
   │   │
   │   └─▶ List[TextSpan]
   │
   ├─▶ Stage 2: LegalClassifier
   │   - Fine-tuned model (if available)
   │   - Rule-based classification
   │   - Semantic validation
   │   │
   │   └─▶ List[LegalClassification]
   │
   ├─▶ Stage 3: NormativeParser
   │   - Regex extraction
   │   - Component parsing
   │   │
   │   └─▶ List[ParsedNormative]
   │
   ├─▶ Stage 4: ReferenceResolver
   │   - Resolution logic
   │   │
   │   └─▶ List[ResolvedNormative]
   │
   └─▶ Stage 5: StructureBuilder
       - Apply label mapping
       - Filter institutions
       - Add metadata
       │
       └─▶ List[Dict[str, Any]]
   │
   ▼
6. Convert to Response Schema
   - Build Entity objects
   - Build LegalSource objects
   - Ensure native Python types
   │
   ▼
7. Return Response
   {
     "entities": [...],
     "legal_sources": [...],
     "metadata": {...}
   }
```

### Request Flow - Active Learning

```
1. Trigger Active Learning
   POST /api/v1/active-learning/start
   {
     "batch_size": 10,
     "strategy": "uncertainty"
   }
   │
   ▼
2. ActiveLearningManager.process_uncertain_documents()
   │
   ├─▶ Query unannotated documents
   │   SELECT * FROM documents
   │   WHERE id NOT IN (SELECT document_id FROM annotation_tasks)
   │
   ├─▶ Run NER pipeline on each document
   │   For doc in documents:
   │     entities = pipeline.extract_legal_sources(doc.text)
   │     uncertainty = calculate_uncertainty(entities)
   │     doc_data.append({doc_id, entities, uncertainty})
   │
   ├─▶ Sort by uncertainty (desc)
   │   sorted_docs = sorted(doc_data, key=lambda x: x["uncertainty"], reverse=True)
   │
   ├─▶ Select top N
   │   selected_docs = sorted_docs[:batch_size]
   │
   └─▶ Create annotation tasks
       For doc in selected_docs:
         │
         ├─▶ Create AnnotationTask
         │   task = AnnotationTask(
         │     document_id=doc["doc_id"],
         │     status="pending",
         │     priority=doc["uncertainty"]
         │   )
         │   db.add(task)
         │
         └─▶ Save pre-extracted entities
             For entity in doc["entities"]:
               │
               ├─▶ Apply label mapping
               │   act_type = entity["act_type"]
               │   label = act_type_to_label(act_type)
               │
               ├─▶ Convert to native types
               │   start_char = int(entity["start_char"])
               │   end_char = int(entity["end_char"])
               │   confidence = float(entity["confidence"])
               │
               └─▶ Save Entity
                   entity = Entity(
                     document_id=doc["doc_id"],
                     text=entity["text"],
                     label=label,
                     start_char=start_char,
                     end_char=end_char,
                     confidence=confidence
                   )
                   db.add(entity)
       │
       db.commit()
   │
   ▼
3. Return Response
   {
     "status": "success",
     "tasks_created": 10,
     "documents_processed": 50
   }
```

### Data Flow - Training

```
1. Trigger Training
   POST /api/v1/active-learning/train
   │
   ▼
2. DatasetBuilder.build_dataset_from_db()
   │
   ├─▶ Query annotated documents
   │   SELECT d.*, e.*, a.*
   │   FROM documents d
   │   JOIN entities e ON e.document_id = d.id
   │   JOIN annotations a ON a.entity_id = e.id
   │   WHERE a.is_correct IS NOT NULL
   │
   ├─▶ Convert to IOB format
   │   For each document:
   │     - Tokenize text
   │     - Assign BIO tags based on entities
   │     - Handle overlaps
   │     - Apply label corrections from annotations
   │
   ├─▶ Split dataset
   │   - Train: 80%
   │   - Validation: 10%
   │   - Test: 10%
   │
   └─▶ Save to MinIO
       minio_client.put_object(
         bucket="legal-ner-datasets",
         object_name=f"dataset_{timestamp}.jsonl",
         data=dataset_bytes
       )
   │
   ▼
3. Fine-tune Model
   │
   ├─▶ Load base model
   │   model = AutoModelForTokenClassification.from_pretrained(
   │     "dlicari/Italian-Legal-BERT"
   │   )
   │
   ├─▶ Prepare training arguments
   │   training_args = TrainingArguments(
   │     output_dir="./models/fine_tuned",
   │     num_train_epochs=3,
   │     per_device_train_batch_size=16,
   │     evaluation_strategy="epoch",
   │     save_strategy="epoch",
   │     load_best_model_at_end=True
   │   )
   │
   ├─▶ Train
   │   trainer = Trainer(
   │     model=model,
   │     args=training_args,
   │     train_dataset=train_dataset,
   │     eval_dataset=val_dataset
   │   )
   │   trainer.train()
   │
   └─▶ Evaluate
       metrics = trainer.evaluate(test_dataset)
   │
   ▼
4. Save Model
   │
   ├─▶ Save to MinIO
   │   model_path = f"models/fine_tuned_{timestamp}"
   │   trainer.save_model(model_path)
   │   # Upload to MinIO
   │
   └─▶ Register in database
       ml_model = MLModel(
         name=f"fine_tuned_{timestamp}",
         version="1.0.0",
         model_path=minio_path,
         metrics=metrics,
         is_active=False  # Manual activation required
       )
       db.add(ml_model)
       db.commit()
   │
   ▼
5. Return Response
   {
     "status": "success",
     "model_id": 123,
     "metrics": {...}
   }
```

---

## Storage e Persistenza

### PostgreSQL Schema

```sql
-- Documents
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    source VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Entities
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
    text VARCHAR NOT NULL,
    label VARCHAR NOT NULL,          -- Standardized label (D.LGS, CODICE_CIVILE, etc.)
    start_char INTEGER NOT NULL,
    end_char INTEGER NOT NULL,
    confidence FLOAT,
    model VARCHAR,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Annotation Tasks
CREATE TABLE annotation_tasks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER REFERENCES documents(id),
    status VARCHAR DEFAULT 'pending',  -- pending, in_progress, completed
    priority FLOAT DEFAULT 0.0,        -- Uncertainty score
    assigned_to VARCHAR,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Annotations (User Feedback)
CREATE TABLE annotations (
    id SERIAL PRIMARY KEY,
    entity_id INTEGER REFERENCES entities(id),
    user_id VARCHAR NOT NULL,
    is_correct BOOLEAN NOT NULL,
    corrected_label VARCHAR,
    feedback_text TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- ML Models
CREATE TABLE ml_models (
    id SERIAL PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    version VARCHAR NOT NULL,
    model_path VARCHAR NOT NULL,      -- MinIO path
    is_active BOOLEAN DEFAULT FALSE,
    metrics JSON,                      -- {precision, recall, f1, accuracy}
    created_at TIMESTAMP DEFAULT NOW(),
    trained_at TIMESTAMP
);

-- Indexes
CREATE INDEX idx_entities_document_id ON entities(document_id);
CREATE INDEX idx_entities_label ON entities(label);
CREATE INDEX idx_annotation_tasks_status ON annotation_tasks(status);
CREATE INDEX idx_annotation_tasks_priority ON annotation_tasks(priority DESC);
CREATE INDEX idx_annotations_entity_id ON annotations(entity_id);
CREATE INDEX idx_ml_models_is_active ON ml_models(is_active);
```

### MinIO Storage Structure

```
legal-ner-datasets/           # Bucket
├── datasets/                 # Training datasets
│   ├── dataset_20250101_120000.jsonl
│   ├── dataset_20250101_130000.jsonl
│   └── ...
│
├── models/                   # Fine-tuned models
│   ├── fine_tuned_20250101_120000/
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.txt
│   │   └── label_config.json
│   └── ...
│
├── exports/                  # Data exports
│   ├── export_20250101_120000.jsonl
│   └── ...
│
└── logs/                     # Archived logs (optional)
    └── ...
```

---

## Scalabilità e Performance

### Current Architecture Limitations

```
Single Instance Setup:
├─ FastAPI: 1 instance
├─ PostgreSQL: 1 instance
├─ MinIO: 1 instance
└─ Limits:
   - ~100 concurrent requests
   - ~1000 documents/hour processing
   - Single point of failure
```

### Scaling Strategy (Future)

```
Horizontal Scaling:
┌──────────────────────────────────────────────┐
│           Load Balancer (Nginx)              │
└──────────────────┬───────────────────────────┘
                   │
         ┌─────────┼─────────┐
         │         │         │
    ┌────▼───┐ ┌──▼────┐ ┌──▼────┐
    │FastAPI │ │FastAPI│ │FastAPI│
    │Instance│ │Instance│ │Instance│
    └────┬───┘ └──┬────┘ └──┬────┘
         │        │         │
         └────────┼─────────┘
                  │
         ┌────────▼─────────┐
         │   PostgreSQL     │
         │   (Read Replicas)│
         └──────────────────┘

Benefits:
- 10x throughput
- High availability
- Zero-downtime deployments
```

### Performance Optimizations

**1. Pipeline Optimization**:
```python
# Cache model loading
_model_cache = {}

def get_model(model_name):
    if model_name not in _model_cache:
        _model_cache[model_name] = load_model(model_name)
    return _model_cache[model_name]
```

**2. Database Optimization**:
```python
# Batch inserts
entities_batch = []
for entity_data in entities:
    entities_batch.append(Entity(**entity_data))

db.bulk_save_objects(entities_batch)
db.commit()
```

**3. Configuration Caching**:
```python
# YAML configs loaded once and cached
_config_cache = None

def get_config():
    global _config_cache
    if _config_cache is None:
        _config_cache = yaml.safe_load(open("config.yaml"))
    return _config_cache
```

---

## Sicurezza

### Authentication & Authorization

**API Key Authentication**:
```python
# Dependencies
async def get_api_key(api_key: str = Header(..., alias="X-API-Key")):
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

# Usage
@router.get("/protected")
async def protected_route(api_key: str = Depends(get_api_key)):
    return {"message": "Access granted"}
```

### Data Security

**1. Database**:
- Connection via SSL (production)
- Credentials in environment variables
- Regular backups

**2. MinIO**:
- Access/Secret keys
- Bucket policies
- Encryption at rest (optional)

**3. Logging**:
- No sensitive data in logs
- Structured logging for audit
- Log rotation

### Input Validation

**Pydantic Models**:
```python
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)

    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        return v
```

---

**Versione**: 1.0.0
**Ultima Modifica**: 2025-10-01
**Autore**: Sistema Legal-NER Team
