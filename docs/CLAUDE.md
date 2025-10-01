# Legal-NER System - Claude Code Guide

> **Guida per Claude Code**: Questo documento fornisce tutte le informazioni necessarie per lavorare efficacemente sul sistema Legal-NER.

## 📋 Indice

1. [Panoramica del Sistema](#panoramica-del-sistema)
2. [Architettura](#architettura)
3. [Struttura del Progetto](#struttura-del-progetto)
4. [Pipeline NER Specializzata](#pipeline-ner-specializzata)
5. [Sistema di Configurazione](#sistema-di-configurazione)
6. [Database e Modelli](#database-e-modelli)
7. [API Endpoints](#api-endpoints)
8. [Sistema di Labeling](#sistema-di-labeling)
9. [Active Learning](#active-learning)
10. [Interfaccia Utente](#interfaccia-utente)
11. [Task Comuni](#task-comuni)
12. [Troubleshooting](#troubleshooting)

---

## 🎯 Panoramica del Sistema

**Legal-NER** è un sistema avanzato di Named Entity Recognition specializzato nell'estrazione e classificazione di riferimenti normativi da testi legali italiani.

### Caratteristiche Principali

- **Pipeline NER Multi-Stage**: 5 stage specializzati per massima precisione
- **Active Learning**: Apprendimento continuo con feedback degli utenti
- **Configurazione YAML**: Sistema completamente configurabile senza modifiche al codice
- **Label Mapping Centralizzato**: Sistema uniforme di etichettatura
- **UI Web Interattiva**: Interfaccia per annotazione e gestione
- **API RESTful**: Endpoint completi per integrazione

### Stack Tecnologico

- **Backend**: FastAPI (Python 3.13)
- **ML/NER**: Transformers (HuggingFace), PyTorch
- **Database**: PostgreSQL + SQLAlchemy ORM
- **Frontend**: Flask + Bootstrap + Jinja2
- **Storage**: MinIO (S3-compatible)
- **Logging**: Structlog (JSON structured logs)

---

## 🏗 Architettura

### Componenti Principali

```
┌─────────────────────────────────────────────────────────────┐
│                      Legal-NER System                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   FastAPI    │  │    Flask     │  │  PostgreSQL  │      │
│  │  (Backend)   │  │     (UI)     │  │  (Database)  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│  ┌──────▼──────────────────▼──────────────────▼───────┐    │
│  │           Application Layer (app/)                  │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │                                                      │    │
│  │  ┌────────────────────────────────────────────┐    │    │
│  │  │      Specialized NER Pipeline              │    │    │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐│    │    │
│  │  │  │ Entity   │→ │  Legal   │→ │ Normative││    │    │
│  │  │  │ Detector │  │Classifier│  │  Parser  ││    │    │
│  │  │  └──────────┘  └──────────┘  └──────────┘│    │    │
│  │  │       ↓              ↓              ↓     │    │    │
│  │  │  ┌──────────┐  ┌──────────────────────┐ │    │    │
│  │  │  │Reference │→ │  Structure Builder   │ │    │    │
│  │  │  │ Resolver │  │                       │ │    │    │
│  │  │  └──────────┘  └──────────────────────┘ │    │    │
│  │  └────────────────────────────────────────────┘    │    │
│  │                                                      │    │
│  │  ┌────────────────────────────────────────────┐    │    │
│  │  │         Active Learning System             │    │    │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐│    │    │
│  │  │  │Uncertainty│→ │   Task   │→ │ Feedback ││    │    │
│  │  │  │ Sampling │  │ Creation │  │Processing││    │    │
│  │  │  └──────────┘  └──────────┘  └──────────┘│    │    │
│  │  └────────────────────────────────────────────┘    │    │
│  │                                                      │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                               │
│  ┌───────────────────────────────────────────────────────┐  │
│  │              Configuration System                      │  │
│  │  • pipeline_config.yaml                               │  │
│  │  • active_learning_config.yaml                        │  │
│  │  • label_mapping.yaml                                 │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Flusso di Elaborazione

```
┌─────────────┐
│   Testo     │
│   Legale    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│   Stage 1: Entity Detector                  │
│   • Identifica span di testo potenziali     │
│   • Modello: Italian-Legal-BERT             │
│   • Output: TextSpan con posizioni precise  │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│   Stage 2: Legal Classifier                 │
│   • Classifica tipo atto normativo          │
│   • Rule-based + Semantic validation        │
│   • Output: LegalClassification             │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│   Stage 3: Normative Parser                 │
│   • Estrae componenti strutturati           │
│   • Numero, data, articolo, comma, etc.     │
│   • Output: ParsedNormative                 │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│   Stage 4: Reference Resolver               │
│   • Risolve riferimenti incompleti          │
│   • Gestisce riferimenti impliciti          │
│   • Output: ResolvedNormative               │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│   Stage 5: Structure Builder                │
│   • Costruisce output finale strutturato    │
│   • Applica label standardizzate            │
│   • Output: Dict[str, Any]                  │
└──────┬──────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│   Entità    │
│  Estratte   │
│  + Metadati │
└─────────────┘
```

---

## 📁 Struttura del Progetto

```
legal-ner/
├── legal-ner-api/              # Backend FastAPI
│   ├── app/
│   │   ├── api/
│   │   │   └── v1/
│   │   │       └── endpoints/  # API endpoints
│   │   │           ├── predict.py          # NER prediction
│   │   │           ├── feedback.py         # User feedback
│   │   │           ├── active_learning.py  # AL management
│   │   │           ├── documents.py        # Document CRUD
│   │   │           ├── annotations.py      # Annotation management
│   │   │           ├── process.py          # Document processing
│   │   │           ├── export.py           # Data export
│   │   │           ├── models.py           # Model management
│   │   │           ├── labels.py           # Label management
│   │   │           └── admin.py            # Admin operations
│   │   │
│   │   ├── core/               # Core utilities
│   │   │   ├── config.py              # Base configuration
│   │   │   ├── config_loader.py       # Pipeline config loader
│   │   │   ├── active_learning_config.py  # AL config loader
│   │   │   ├── label_mapping.py       # Label mapping system
│   │   │   ├── dependencies.py        # FastAPI dependencies
│   │   │   ├── logging.py             # Structured logging
│   │   │   └── model_manager.py       # ML model management
│   │   │
│   │   ├── services/           # Business logic
│   │   │   └── specialized_pipeline.py  # 5-stage NER pipeline
│   │   │
│   │   ├── feedback/           # Active Learning
│   │   │   ├── active_learning.py     # AL manager
│   │   │   └── dataset_builder.py     # Dataset generation
│   │   │
│   │   ├── database/           # Data layer
│   │   │   ├── database.py     # DB connection
│   │   │   ├── models.py       # SQLAlchemy models
│   │   │   └── crud.py         # CRUD operations
│   │   │
│   │   ├── ui/                 # Flask UI
│   │   │   ├── app.py          # Flask application
│   │   │   └── templates/      # Jinja2 templates
│   │   │       ├── index.html
│   │   │       ├── annotate.html
│   │   │       ├── admin.html
│   │   │       └── feedback.html
│   │   │
│   │   └── main.py             # FastAPI entry point
│   │
│   ├── config/                 # Configuration files
│   │   ├── pipeline_config.yaml        # Pipeline configuration
│   │   ├── active_learning_config.yaml # Active learning config
│   │   └── label_mapping.yaml          # Label mappings
│   │
│   ├── data/                   # Data storage (MinIO)
│   ├── logs/                   # Application logs
│   ├── tests/                  # Test suite
│   ├── requirements.txt        # Python dependencies
│   └── .env                    # Environment variables
│
├── docs/                       # Documentation
│   ├── CLAUDE.md              # This file
│   ├── ARCHITECTURE.md        # System architecture
│   ├── CONFIGURATION.md       # Configuration guide
│   ├── API_REFERENCE.md       # API documentation
│   └── LABEL_SYSTEM.md        # Label system guide
│
└── README.md                   # Project overview
```

---

## 🔬 Pipeline NER Specializzata

### File: `app/services/specialized_pipeline.py`

La pipeline è composta da 5 stage specializzati, ognuno con un ruolo specifico:

### Stage 1: EntityDetector

**Responsabilità**: Identificare span di testo che potrebbero essere riferimenti normativi

**Modelli**:
- Primary: `dlicari/Italian-Legal-BERT`
- Fallback: `Babelscape/wikineural-multilingual-ner`

**Caratteristiche**:
- Tokenization con offset mapping per posizioni precise
- Rule-based fallback se il modello non trova entità
- Espansione automatica dei confini per catturare riferimenti completi
- Filtro anti-spurio configurabile

**Metodi Principali**:
```python
def detect_candidates(text: str, log_file_path: Optional[str] = None) -> List[TextSpan]
```

### Stage 2: LegalClassifier

**Responsabilità**: Classificare il tipo di atto normativo

**Approcci**:
1. **Rule-based**: Pattern regex con confidence configurabile
2. **Semantic**: Embeddings + similarity con prototipi
3. **Fine-tuned**: Modello addestrato (opzionale)

**Strategia**:
- Rule-based ha priorità per pattern affidabili (es. "c.c." = 0.99 confidence)
- Validazione semantica se confidence bassa o sempre (configurabile)
- Gestione discrepanze tra rule-based e semantic

**Metodi Principali**:
```python
def classify_legal_type(text_span: TextSpan, context: str, log_file_path: Optional[str] = None) -> LegalClassification
```

### Stage 3: NormativeParser

**Responsabilità**: Estrarre componenti strutturati

**Componenti Estratti**:
- `act_type`: Tipo di atto
- `act_number`: Numero dell'atto
- `date`: Data (formato vario)
- `article`: Articolo
- `comma`: Comma
- `letter`: Lettera
- `version`: Versione
- `annex`: Allegato

**Metodi Principali**:
```python
def parse(legal_classification: LegalClassification) -> ParsedNormative
```

### Stage 4: ReferenceResolver

**Responsabilità**: Risolvere riferimenti incompleti o ambigui

**Funzionalità** (attualmente semplificata):
- Risoluzione diretta
- Futuro: gestione riferimenti impliciti, catene di riferimenti

**Metodi Principali**:
```python
def resolve(parsed_normative: ParsedNormative, full_text: str) -> ResolvedNormative
```

### Stage 5: StructureBuilder

**Responsabilità**: Costruire output finale strutturato

**Funzionalità**:
- Filtro istituzioni (opzionale)
- Filtro valori null (opzionale)
- Normalizzazione abbreviazioni
- Metadati aggiuntivi

**Metodi Principali**:
```python
def build(resolved_normative: ResolvedNormative) -> Dict[str, Any]
```

### Conversione Tipi Python Nativi

**IMPORTANTE**: La pipeline converte automaticamente tutti i tensori PyTorch e tipi numpy a tipi Python nativi per compatibilità con PostgreSQL:

```python
# start_char, end_char: torch.Tensor → int
# confidence: np.float32 → float
```

Questa conversione avviene in:
- `_extract_entities_with_offsets()`
- `_expand_reference_boundaries()`
- Output finale della pipeline

---

## ⚙️ Sistema di Configurazione

### File di Configurazione

#### 1. `config/pipeline_config.yaml`

Controlla TUTTI gli aspetti della pipeline NER:

**Sezioni Principali**:

```yaml
# Modelli AI
models:
  entity_detector:
    primary: "dlicari/Italian-Legal-BERT"
    fallback: "Babelscape/wikineural-multilingual-ner"
    max_length: 256

  legal_classifier:
    primary: "dlicari/distil-ita-legal-bert"

# Soglie di confidence
confidence_thresholds:
  minimum_detection_confidence: 0.5
  rule_based_priority_threshold: 0.8
  semantic_similarity_scale: 0.95

  # Confidence specifiche per ogni tipo di atto
  rule_based_confidence:
    codice_civile_abbrev: 0.99
    decreto_legislativo_full: 0.98
    # ... (100+ configurazioni)

# Finestre di contesto
context_windows:
  entity_expansion:
    left_window: 150
    right_window: 150

  semantic_context:
    immediate_context: 50
    extended_context: 120

# Mappatura NORMATTIVA (abbreviazioni → act_type)
normattiva_mapping:
  decreto_legislativo:
    - "d.lgs."
    - "d.lgs"
    - "decreto legislativo"

  codice_civile:
    - "c.c."
    - "cc"
    - "codice civile"
  # ... (500+ pattern)

# Pattern regex per rilevamento
regex_patterns:
  legal_acts:
    - r'\b(?:d\.?\s*lgs\.?|decreto\s+legislativo)\s+n?\.?\s*\d+(?:/\d{4})?'
    # ... (100+ pattern)

  codes:
    - r'\bc\.?\s*c\.?(?:\s+|$)'  # c.c.
    # ... (50+ pattern)

# Pattern contestuali
context_patterns:
  normative_references:
    - r'(?:secondo|ai\s+sensi|in\s+base\s+a)'
    # ... (50+ pattern)

# Espansione confini
boundary_expansion:
  left_patterns:
    - r'(decreto\s+legislativo\s+n?\.?\s?)$'
    # ... (100+ pattern)

  right_patterns:
    - r'^(\s+del\s+\d{4})'
    # ... (50+ pattern)

# Prototipi semantici per classificazione
semantic_prototypes:
  decreto_legislativo:
    - "decreto legislativo numero del anno"
    - "d.lgs. n. del"
  # ... (300+ prototipi)

# Pattern di parsing
parsing_patterns:
  act_number: r"(?:n\.?|numero)\s*(\d+)"
  date: r"(?:del|in\s+data\s+del)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4})"
  # ... (20+ pattern)

# Filtri anti-spurio
spurious_filters:
  min_length: 3
  valid_short_terms: ["l.", "cc", "c.c.", ...]
  spurious_words: ["il", "la", "del", ...]
  spurious_patterns:
    - r"^s\.\s*$"
    # ... (20+ pattern)

# Parole contesto legale
legal_context_words:
  - "decreto"
  - "legge"
  - "articolo"
  # ... (100+ parole)

# Configurazione output
output_settings:
  filter_institutions: true
  filter_null_values: true
  quality_checks:
    min_confidence_threshold: 0.6
```

**Caricamento**:
```python
from app.core.config_loader import get_pipeline_config

config = get_pipeline_config()
```

#### 2. `config/active_learning_config.yaml`

Controlla il sistema di active learning:

```yaml
# Active Learning
active_learning:
  selection_strategy: "uncertainty"  # uncertainty, random, diversity
  uncertainty_threshold: 0.7
  default_batch_size: 10
  use_fine_tuned_if_available: true
  high_confidence_threshold: 0.9
  fine_tuned_confidence_threshold: 0.7

# Dataset
dataset:
  min_document_length: 20
  max_sequence_length: 512
  min_entities_per_document: 1
  require_validation: true

# Feedback Loop
feedback_loop:
  enabled: true
  check_interval_seconds: 3600
  auto_training_threshold: 100  # Numero di annotazioni per auto-training
  primary_evaluation_metric: "f1_score"

# Labels (BIO encoding)
labels:
  label_list:
    - O
    - B-codice_civile
    - I-codice_civile
    - B-decreto_legislativo
    - I-decreto_legislativo
    # ... (100+ label)

  label2id:
    O: 0
    B-codice_civile: 1
    # ...

  id2label:
    0: O
    1: B-codice_civile
    # ...
```

**Caricamento**:
```python
from app.core.active_learning_config import get_active_learning_config

config = get_active_learning_config()
```

#### 3. `config/label_mapping.yaml`

Mappatura centralizzata act_type → label standardizzata:

```yaml
# Mappatura act_type → label
act_type_to_label:
  # Decreti
  decreto_legislativo: D.LGS
  decreto_legge: D.L
  decreto_presidente_repubblica: D.P.R
  decreto_ministeriale: D.M
  dpcm: D.P.C.M

  # Leggi
  legge: LEGGE
  legge_costituzionale: LEGGE_COST
  legge_regionale: L.R

  # Codici
  codice_civile: CODICE_CIVILE
  codice_penale: CODICE_PENALE
  codice_procedura_civile: CODICE_PROCEDURA_CIVILE
  codice_crisi_impresa: CODICE_CRISI_IMPRESA

  # Testi Unici
  testo_unico: T.U
  testo_unico_bancario: T.U.B
  testo_unico_enti_locali: T.U.E.L

  # Normativa UE
  direttiva_ue: DIR_UE
  regolamento_ue: REG_UE

  # Altro
  costituzione: COSTITUZIONE
  unknown: UNKNOWN

# Categorie per organizzazione UI
label_categories:
  Decreti:
    - D.LGS
    - D.L
    - D.P.R
    - D.M
    - D.P.C.M

  Leggi:
    - LEGGE
    - LEGGE_COST
    - L.R

  Codici:
    - CODICE_CIVILE
    - CODICE_PENALE
    - CODICE_PROCEDURA_CIVILE

  Testi Unici:
    - T.U
    - T.U.B
    - T.U.E.L

  Normativa UE:
    - DIR_UE
    - REG_UE

  Costituzione:
    - COSTITUZIONE

  Altro:
    - UNKNOWN
```

**Utilizzo**:
```python
from app.core.label_mapping import act_type_to_label, get_all_labels

label = act_type_to_label("decreto_legislativo")  # → "D.LGS"
all_labels = get_all_labels()  # Lista di tutte le label disponibili
```

### Variabili d'Ambiente (.env)

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/legal_ner

# MinIO (S3-compatible storage)
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=legal-ner-datasets
MINIO_USE_SSL=False

# API
API_KEY=your-secure-api-key-here
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000

# Flask UI
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_SECRET_KEY=your-secret-key-here

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

---

## 🗄 Database e Modelli

### File: `app/database/models.py`

#### Modelli SQLAlchemy

**1. Document**
```python
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text, nullable=False)
    source = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    entities = relationship("Entity", back_populates="document", cascade="all, delete-orphan")
    annotation_tasks = relationship("AnnotationTask", back_populates="document")
```

**2. Entity**
```python
class Entity(Base):
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"))
    text = Column(String, nullable=False)
    label = Column(String, nullable=False)  # Label standardizzata (es. "D.LGS")
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=True)
    model = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="entities")
    annotations = relationship("Annotation", back_populates="entity")
```

**3. AnnotationTask**
```python
class AnnotationTask(Base):
    __tablename__ = "annotation_tasks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    status = Column(String, default="pending")  # pending, in_progress, completed
    priority = Column(Float, default=0.0)
    assigned_to = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    document = relationship("Document", back_populates="annotation_tasks")
```

**4. Annotation**
```python
class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(Integer, ForeignKey("entities.id"))
    user_id = Column(String, nullable=False)
    is_correct = Column(Boolean, nullable=False)
    corrected_label = Column(String, nullable=True)
    feedback_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    entity = relationship("Entity", back_populates="annotations")
```

**5. MLModel**
```python
class MLModel(Base):
    __tablename__ = "ml_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    version = Column(String, nullable=False)
    model_path = Column(String, nullable=False)  # MinIO path
    is_active = Column(Boolean, default=False)
    metrics = Column(JSON, nullable=True)  # Accuracy, F1, etc.
    created_at = Column(DateTime, default=datetime.utcnow)
    trained_at = Column(DateTime, nullable=True)
```

### CRUD Operations

File: `app/database/crud.py`

```python
# Create document
def create_document(db: Session, text: str) -> models.Document

# Create entities for document
def create_entities_for_document(db: Session, document_id: int, entities: list[schemas.Entity])

# Create annotations
def create_annotations(db: Session, annotations: list[schemas.Annotation], user_id: str)

# Create annotation task
def create_annotation_task(db: Session, document_id: int, priority: float = 0.0) -> models.AnnotationTask
```

---

## 🔌 API Endpoints

### Base URL: `http://localhost:8000/api/v1`

#### NER & Prediction

**POST /predict**
```json
{
  "text": "Secondo l'art. 2043 c.c., chiunque cagiona danno deve risarcirlo."
}
```

Response:
```json
{
  "entities": [
    {
      "text": "art. 2043 c.c.",
      "label": "CODICE_CIVILE",
      "start_char": 12,
      "end_char": 26,
      "confidence": 0.99,
      "source_type": "codice_civile",
      "act_number": null,
      "article": "2043"
    }
  ],
  "legal_sources": [...]
}
```

#### Documents

**GET /documents**
- Lista tutti i documenti

**GET /documents/{document_id}**
- Ottieni documento specifico

**POST /documents**
```json
{
  "text": "Testo del documento",
  "source": "Fonte opzionale"
}
```

**DELETE /documents/{document_id}**
- Elimina documento

#### Annotations

**GET /annotations/tasks**
- Lista task di annotazione

**GET /annotations/tasks/{task_id}**
- Dettagli task specifico

**POST /annotations/submit**
```json
{
  "task_id": 123,
  "user_id": "user@example.com",
  "annotations": [
    {
      "entity_id": 456,
      "is_correct": true,
      "corrected_label": null
    }
  ]
}
```

#### Feedback

**POST /feedback**
```json
{
  "entity_id": 456,
  "user_id": "user@example.com",
  "is_correct": false,
  "corrected_label": "DECRETO_LEGISLATIVO",
  "feedback_text": "Errore di classificazione"
}
```

**GET /feedback/stats**
- Statistiche feedback

#### Active Learning

**POST /active-learning/start**
```json
{
  "batch_size": 10,
  "strategy": "uncertainty"
}
```

**GET /active-learning/status**
- Stato del sistema AL

**POST /active-learning/train**
- Avvia training con feedback raccolto

#### Labels

**GET /labels**
- Lista tutte le label standardizzate

**GET /labels/categories**
- Label organizzate per categoria

**GET /labels/mapping**
- Mappatura completa act_type → label

**POST /labels/reload**
- Ricarica configurazione label

#### Models

**GET /models**
- Lista tutti i modelli ML

**GET /models/{model_id}**
- Dettagli modello specifico

**POST /models/activate/{model_id}**
- Attiva modello specifico

**GET /models/active**
- Ottieni modello attivo

#### Admin

**POST /admin/reprocess-tasks**
```json
{
  "task_ids": [123, 456, 789],
  "replace_existing": true
}
```

**GET /admin/stats**
- Statistiche sistema

#### Export

**GET /export/dataset**
- Esporta dataset in formato IOB/CoNLL

**GET /export/annotations**
- Esporta annotazioni

---

## 🏷 Sistema di Labeling

### File: `app/core/label_mapping.py`

Il sistema di labeling centralizzato garantisce coerenza tra:
- Pipeline NER (act_type interno)
- Database (label standardizzata)
- UI (visualizzazione)
- API (risposta)

### Conversione Act Type → Label

```python
from app.core.label_mapping import act_type_to_label

# Esempi
act_type_to_label("decreto_legislativo")  # → "D.LGS"
act_type_to_label("codice_civile")        # → "CODICE_CIVILE"
act_type_to_label("legge")                # → "LEGGE"
act_type_to_label("unknown")              # → "UNKNOWN"
```

### Conversione Label → Act Type

```python
from app.core.label_mapping import label_to_act_type

label_to_act_type("D.LGS")           # → "decreto_legislativo"
label_to_act_type("CODICE_CIVILE")  # → "codice_civile"
```

### Funzioni Utility

```python
from app.core.label_mapping import (
    get_all_labels,           # Lista tutte le label
    get_label_categories,     # Categorie di label
    get_label_category,       # Categoria di una label
    validate_label,           # Valida una label
    update_label_mapping,     # Aggiorna mappatura
    reload_label_config       # Ricarica configurazione
)
```

### Dove Viene Usato

1. **admin.py**: Reprocessing task
```python
from app.core.label_mapping import act_type_to_label as convert_act_type_to_label

act_type = entity_data.get("act_type", "unknown")
label = convert_act_type_to_label(act_type)
```

2. **process.py**: Processamento documenti
3. **active_learning.py**: Creazione task
4. **crud.py**: Operazioni database
5. **annotations.py**: Gestione annotazioni

---

## 🎓 Active Learning

### File: `app/feedback/active_learning.py`

### ActiveLearningManager

Gestisce il ciclo completo di active learning:

**Flusso**:
1. **Selezione**: Identifica documenti con alta incertezza
2. **Task Creation**: Crea task di annotazione
3. **Feedback Collection**: Raccoglie feedback utenti
4. **Training**: Addestra nuovo modello con feedback

### Metodi Principali

```python
class ActiveLearningManager:
    async def process_uncertain_documents(
        self,
        batch_size: int = 10,
        strategy: str = "uncertainty"
    ) -> Dict[str, Any]:
        """
        Processa documenti con alta incertezza e crea task.

        Strategy:
        - uncertainty: Seleziona documenti con confidence bassa
        - random: Selezione casuale
        - diversity: Massimizza diversità (TODO)
        """

    async def train_from_feedback(
        self,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Addestra nuovo modello usando feedback raccolto.

        Steps:
        1. Export dataset da database
        2. Prepara dataset training (IOB format)
        3. Fine-tune modello base
        4. Valida su test set
        5. Salva modello in MinIO
        6. Registra in database
        """
```

### Strategia di Uncertainty Sampling

```python
def calculate_uncertainty(entities: List[Dict]) -> float:
    """
    Calcola incertezza media delle entità in un documento.

    Uncertainty = 1 - avg(confidence)

    Documenti con alta uncertainty sono prioritari per annotazione.
    """
    if not entities:
        return 1.0

    confidences = [e.get("confidence", 0.0) for e in entities]
    avg_confidence = sum(confidences) / len(confidences)

    return 1.0 - avg_confidence
```

### Dataset Builder

File: `app/feedback/dataset_builder.py`

```python
class DatasetBuilder:
    def build_dataset_from_db(
        self,
        db: Session,
        output_path: str,
        format: str = "iob",
        include_unannotated: bool = False
    ) -> Dict[str, Any]:
        """
        Costruisce dataset da database.

        Formati supportati:
        - iob: Token-level IOB tagging
        - json: Structured JSON
        - conll: CoNLL format
        """

    def convert_to_iob_format(
        self,
        documents: List[Document]
    ) -> List[Dict]:
        """
        Converte documenti in formato IOB.

        Output:
        {
            "tokens": ["secondo", "l'art.", "2043", "c.c."],
            "tags": ["O", "B-CODICE_CIVILE", "I-CODICE_CIVILE", "I-CODICE_CIVILE"]
        }
        """
```

---

## 🖥 Interfaccia Utente

### File: `app/ui/app.py` (Flask)

### Pagine Principali

#### 1. Home (`/`)
- Dashboard con statistiche
- Accesso rapido a funzionalità principali

#### 2. Annotazione (`/annotate`)
- Interfaccia per annotare task
- Visualizzazione entità con highlighting
- Selezione testo e assegnazione label
- Correzione entità esistenti

**Features**:
- Highlighting colorato per confidence
- Selezione testo per nuove entità
- Dropdown label dinamico
- Feedback testuale opzionale

#### 3. Feedback (`/feedback`)
- Statistiche feedback raccolto
- Lista annotazioni per documento
- Filtri per stato, utente, label

#### 4. Admin (`/admin`)
- Gestione task di annotazione
- Reprocessing documenti
- Statistiche sistema
- Gestione modelli

**Features**:
- Reprocess singolo task o batch
- Visualizzazione log processamento
- Controllo stato pipeline

### Templates (Jinja2)

```
templates/
├── base.html           # Template base con navbar
├── index.html          # Home page
├── annotate.html       # Interfaccia annotazione
├── feedback.html       # Statistiche feedback
└── admin.html          # Admin panel
```

### Styling

- **Framework**: Bootstrap 5
- **Icons**: Bootstrap Icons
- **Custom CSS**: Inline per highlighting entità

---

## 🛠 Task Comuni

### 1. Aggiungere una Nuova Label

**Step 1**: Aggiorna `config/label_mapping.yaml`
```yaml
act_type_to_label:
  nuovo_tipo_atto: NUOVA_LABEL

label_categories:
  Categoria_Appropriata:
    - NUOVA_LABEL
```

**Step 2**: Aggiorna `config/pipeline_config.yaml`
```yaml
normattiva_mapping:
  nuovo_tipo_atto:
    - "abbreviazione"
    - "forma completa"

confidence_thresholds:
  rule_based_confidence:
    nuovo_tipo_atto: 0.95
```

**Step 3**: Ricarica configurazione
```bash
curl -X POST http://localhost:8000/api/v1/labels/reload \
  -H "X-API-Key: your-api-key"
```

### 2. Modificare Confidence di un Pattern

**File**: `config/pipeline_config.yaml`

```yaml
confidence_thresholds:
  rule_based_confidence:
    codice_civile_abbrev: 0.99  # Cambia qui
```

**Note**: Le modifiche sono applicate immediatamente al prossimo caricamento della pipeline.

### 3. Aggiungere Pattern Regex

**File**: `config/pipeline_config.yaml`

```yaml
regex_patterns:
  legal_acts:
    - r'\bnuovo\s+pattern\s+\d+'  # Aggiungi qui
```

### 4. Eseguire Active Learning

**Step 1**: Processa documenti incerti
```bash
curl -X POST http://localhost:8000/api/v1/active-learning/start \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"batch_size": 10, "strategy": "uncertainty"}'
```

**Step 2**: Annota task (via UI o API)

**Step 3**: Avvia training
```bash
curl -X POST http://localhost:8000/api/v1/active-learning/train \
  -H "X-API-Key: your-api-key"
```

### 5. Debug della Pipeline

**Abilita logging dettagliato**:

File: `config/pipeline_config.yaml`
```yaml
output_settings:
  enable_debug_logging: true
  log_pattern_matches: true
```

**Logs**: I log dettagliati sono salvati in `logs/requests/{request_id}.log` in formato JSON.

**Analisi log**:
```bash
# Trova log di una richiesta specifica
ls -lt logs/requests/ | head -n 5

# Analizza log
cat logs/requests/{request_id}.log | jq .
```

### 6. Backup Database

```bash
# Backup PostgreSQL
pg_dump -U user -h localhost legal_ner > backup.sql

# Restore
psql -U user -h localhost legal_ner < backup.sql
```

### 7. Export Dataset

```bash
# Via API
curl -X GET http://localhost:8000/api/v1/export/dataset \
  -H "X-API-Key: your-api-key" \
  -o dataset.jsonl

# Via UI
# Naviga a http://localhost:5000/admin → Export Dataset
```

---

## 🐛 Troubleshooting

### Errore: "can't adapt type 'Tensor'"

**Causa**: Valori PyTorch Tensor o numpy non convertiti a tipi Python nativi.

**Soluzione**: Il sistema ora converte automaticamente. Se l'errore persiste, verifica che:
1. `specialized_pipeline.py` converta tutti i valori nell'output
2. Gli endpoint convertano valori prima di inserire in DB

**Debug**:
```python
import torch
import numpy as np

# Verifica tipo
value = result.get("start_char")
print(type(value))  # Deve essere <class 'int'>, non torch.Tensor

# Conversione manuale se necessario
if isinstance(value, (torch.Tensor, np.integer)):
    value = int(value.item() if hasattr(value, 'item') else value)
```

### Errore: Label non trovata

**Causa**: Discrepanza tra act_type della pipeline e label_mapping.

**Soluzione**:
1. Verifica `config/label_mapping.yaml` contenga l'act_type
2. Esegui reload: `POST /api/v1/labels/reload`
3. Controlla log per act_type mancanti

**Fallback**: Il sistema usa `act_type.upper()` se la mappatura non esiste.

### Pipeline restituisce 0 entità

**Possibili cause**:
1. **Threshold troppo alto**: Abbassa `minimum_detection_confidence`
2. **Pattern non riconosciuti**: Aggiungi pattern in `normattiva_mapping`
3. **Modello non caricato**: Verifica log startup

**Debug**:
```python
# Test pipeline con logging
from app.services.specialized_pipeline import LegalSourceExtractionPipeline

pipeline = LegalSourceExtractionPipeline()
text = "Test: art. 2043 c.c."
log_path = "debug.log"

results = await pipeline.extract_legal_sources(text, log_file_path=log_path)

# Analizza log
with open(log_path) as f:
    for line in f:
        print(json.loads(line))
```

### Modelli Transformer non si caricano

**Errore**: `OSError: Can't load model`

**Soluzioni**:
1. **Verifica connessione internet** (primo caricamento scarica da HuggingFace)
2. **Cache corrotta**: Elimina `~/.cache/huggingface/`
3. **Fallback**: La pipeline usa modello fallback se primary fallisce

**Force reload**:
```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("dlicari/Italian-Legal-BERT", force_download=True)
```

### Active Learning non crea task

**Cause possibili**:
1. **Threshold troppo alto**: Abbassa `uncertainty_threshold` in config
2. **Nessun documento incerto**: Tutti i documenti hanno alta confidence
3. **Database vuoto**: Importa documenti prima

**Debug**:
```python
from app.feedback.active_learning import ActiveLearningManager

manager = ActiveLearningManager(pipeline)
result = await manager.process_uncertain_documents(batch_size=10)

print(result)
# Output: {"status": "success", "tasks_created": N, "documents_processed": M}
```

### UI non mostra entità

**Verifica**:
1. **API risponde**: `curl http://localhost:8000/health`
2. **CORS configurato**: Verifica `main.py` include origin Flask
3. **Console browser**: Apri DevTools → Console per errori JS

**Fix CORS**:
```python
# main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000"],  # Aggiungi origin Flask
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### MinIO connection failed

**Errore**: `urllib3.exceptions.MaxRetryError`

**Soluzioni**:
1. **Verifica MinIO running**: `docker ps | grep minio`
2. **Verifica credenziali**: Controlla `.env`
3. **Crea bucket manualmente**: `mc mb minio/legal-ner-datasets`

**Test connessione**:
```python
from minio import Minio

client = Minio(
    "localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)

# Test
buckets = client.list_buckets()
print(buckets)
```

---

## 📚 Risorse Utili

### Documentazione Esterna

- **FastAPI**: https://fastapi.tiangolo.com/
- **Transformers**: https://huggingface.co/docs/transformers/
- **SQLAlchemy**: https://docs.sqlalchemy.org/
- **Structlog**: https://www.structlog.org/
- **MinIO**: https://min.io/docs/minio/linux/index.html

### File di Documentazione del Progetto

- `docs/ARCHITECTURE.md`: Architettura dettagliata
- `docs/CONFIGURATION.md`: Guida configurazione completa
- `docs/API_REFERENCE.md`: Riferimento API completo
- `docs/LABEL_SYSTEM.md`: Sistema di labeling
- `README.md`: Overview generale

### Comandi Rapidi

```bash
# Start FastAPI
cd legal-ner-api
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start Flask UI
cd legal-ner-api
python app/ui/app.py

# Run tests
pytest tests/

# Format code
black app/
isort app/

# Type check
mypy app/

# Lint
flake8 app/

# Generate requirements
pip freeze > requirements.txt
```

---

## ✅ Checklist Pre-Deployment

- [ ] Tutti i test passano
- [ ] Configurazioni validate
- [ ] Database migrato
- [ ] MinIO configurato e testato
- [ ] Variabili d'ambiente settate
- [ ] Logs configurati
- [ ] API key sicure
- [ ] CORS configurato correttamente
- [ ] Backup database schedulato
- [ ] Monitoring configurato
- [ ] Documentazione aggiornata

---

## 🔄 Workflow Tipico

1. **Setup Iniziale**
   ```bash
   # Clone repo
   git clone <repo>
   cd legal-ner

   # Setup virtual environment
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # .venv\Scripts\activate   # Windows

   # Install dependencies
   pip install -r requirements.txt

   # Setup .env
   cp .env.example .env
   # Edit .env with your values

   # Start database
   docker-compose up -d postgres

   # Run migrations
   alembic upgrade head

   # Start MinIO
   docker-compose up -d minio
   ```

2. **Development**
   ```bash
   # Start FastAPI (auto-reload)
   uvicorn app.main:app --reload

   # Start Flask UI
   python app/ui/app.py

   # Run tests
   pytest tests/ -v
   ```

3. **Testing Changes**
   ```bash
   # Test API endpoint
   curl -X POST http://localhost:8000/api/v1/predict \
     -H "Content-Type: application/json" \
     -H "X-API-Key: your-key" \
     -d '{"text": "art. 2043 c.c."}'

   # Check logs
   tail -f logs/app.log
   ```

4. **Deploy**
   ```bash
   # Build Docker images
   docker-compose build

   # Start all services
   docker-compose up -d

   # Verify
   curl http://localhost:8000/health
   ```

---

**Versione**: 1.0.0
**Ultima Modifica**: 2025-10-01
**Autore**: Sistema Legal-NER Team
**Contatto**: [Inserire contatto]

---

*Questa guida è pensata specificamente per Claude Code e viene mantenuta aggiornata ad ogni modifica significativa del sistema.*
