# Legal-NER Configuration Guide

> **Guida Completa alla Configurazione**: Tutti i parametri configurabili del sistema Legal-NER.

## Indice

1. [Overview](#overview)
2. [Environment Variables](#environment-variables)
3. [Pipeline Configuration](#pipeline-configuration)
4. [Active Learning Configuration](#active-learning-configuration)
5. [Label Mapping Configuration](#label-mapping-configuration)
6. [Database Configuration](#database-configuration)
7. [Best Practices](#best-practices)

---

## Overview

Il sistema Legal-NER è completamente configurabile tramite:
- **File YAML**: Configurazioni strutturate per pipeline, active learning, label mapping
- **Environment Variables**: Credenziali e configurazioni deployment
- **Database**: Stato runtime e modelli attivi

**Filosofia**: *Configuration over Code* - Modificare comportamento senza toccare codice.

---

## Environment Variables

File: `.env` (root directory)

### Database

```bash
# PostgreSQL Connection
DATABASE_URL=postgresql://user:password@localhost:5432/legal_ner

# Connection Pool
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
DB_POOL_TIMEOUT=30
DB_POOL_RECYCLE=3600
```

### MinIO (Object Storage)

```bash
# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_USE_SSL=False
MINIO_BUCKET_NAME=legal-ner-datasets

# Bucket auto-creation
MINIO_AUTO_CREATE_BUCKET=True
```

### API Security

```bash
# API Key for authentication
API_KEY=your-secure-random-api-key-here-min-32-chars

# CORS Origins (comma-separated)
CORS_ORIGINS=http://localhost:5000,http://127.0.0.1:5000
```

### Application

```bash
# FastAPI
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
FASTAPI_WORKERS=4
FASTAPI_RELOAD=False  # True in dev, False in prod

# Flask UI
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_SECRET_KEY=your-flask-secret-key-here
FLASK_DEBUG=False  # True in dev, False in prod
```

### Logging

```bash
# Log Configuration
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json  # json, text
LOG_FILE=logs/app.log
LOG_ROTATION=daily  # daily, size-based
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=30
```

### ML Models

```bash
# HuggingFace Cache
HF_HOME=/path/to/huggingface/cache
TRANSFORMERS_CACHE=/path/to/transformers/cache

# Model Download
HF_OFFLINE=False  # True per usare solo cache locale
```

---

## Pipeline Configuration

File: `config/pipeline_config.yaml`

### Section 1: Models

```yaml
models:
  entity_detector:
    primary: "dlicari/Italian-Legal-BERT"
    fallback: "Babelscape/wikineural-multilingual-ner"
    max_length: 256  # Max tokens for tokenization

  legal_classifier:
    primary: "dlicari/distil-ita-legal-bert"
    embedding_max_length: 256

  semantic_correlator:
    model: "dlicari/Italian-Legal-BERT"
```

**Quando modificare**:
- Usare modelli diversi
- Ottimizzare per velocità (modelli più piccoli)
- Ottimizzare per accuratezza (modelli più grandi)

### Section 2: Confidence Thresholds

```yaml
confidence_thresholds:
  # Stage 1: Entity Detection
  minimum_detection_confidence: 0.5  # Min confidence per accettare entità

  # Stage 2: Legal Classification
  rule_based_priority_threshold: 0.8  # Sotto questa soglia, valida con semantic
  semantic_similarity_scale: 0.95  # Scala similarity a confidence
  minimum_classification_confidence: 0.7  # Min confidence finale
  enable_semantic_validation_always: true  # Sempre valida con semantic

  # Gestione discrepanze rule-based vs semantic
  semantic_discrepancy_confidence_threshold: 0.90

  # Confidence per ogni pattern rule-based
  rule_based_confidence:
    # Codici (altissima affidabilità)
    codice_civile_abbrev: 0.99  # "c.c."
    codice_penale_abbrev: 0.99  # "c.p."
    codice_procedura_civile_abbrev: 0.99  # "c.p.c."

    # Decreti
    decreto_legislativo_full: 0.98  # "decreto legislativo"
    decreto_legislativo_abbrev: 0.95  # "d.lgs."

    # Leggi
    legge_full: 0.90
    legge_abbrev: 0.75  # Solo "l." è ambiguo

    # Default
    default: 0.5
```

**Tuning Tips**:
- **Troppi falsi positivi**: Aumentare `minimum_detection_confidence`
- **Troppi falsi negativi**: Diminuire threshold
- **Classificazione errata**: Abilitare `enable_semantic_validation_always`
- **Pattern specifici**: Modificare `rule_based_confidence.{pattern}`

### Section 3: Context Windows

```yaml
context_windows:
  entity_expansion:
    left_window: 150  # Caratteri a sinistra per expansion
    right_window: 150  # Caratteri a destra
    context_window: 75  # Caratteri per context in output

  semantic_context:
    immediate_context: 50  # Contesto immediato
    extended_context: 120  # Contesto esteso
    full_context: 250  # Contesto completo

  classification_context: 200  # Caratteri per classificazione semantica
```

**Tuning Tips**:
- **Riferimenti incompleti**: Aumentare `entity_expansion.left_window/right_window`
- **Performance**: Diminuire tutte le finestre
- **Accuratezza semantica**: Aumentare `classification_context`

### Section 4: NORMATTIVA Mapping

```yaml
normattiva_mapping:
  decreto_legislativo:
    - "d.lgs."
    - "d.lgs"
    - "dlgs"
    - "d.lg."
    - "decreto legislativo"

  codice_civile:
    - "c.c."
    - "cc"
    - "codice civile"

  # ... 500+ mappings
```

**Aggiungere nuovo tipo**:
```yaml
  nuovo_tipo_atto:
    - "abbreviazione"
    - "forma completa"
    - "variante"
```

### Section 5: Regex Patterns

```yaml
regex_patterns:
  legal_acts:
    - r'\b(?:d\.?\s*lgs\.?|decreto\s+legislativo)\s+n?\.?\s*\d+(?:/\d{4})?'
    # Pattern: "d.lgs. 123/2023" o "decreto legislativo n. 123"

  codes:
    - r'\bc\.?\s*c\.?(?:\s+|$)'  # c.c.
    - r'\bc\.?\s*p\.?(?:\s+|$)(?!\.?c\.?|\.?p\.?)'  # c.p. (escludi c.p.c.)

  # ... 200+ patterns
```

**Aggiungere pattern**:
```yaml
  # Sotto categoria appropriata
  legal_acts:
    - r'\bnuovo\s+pattern\s+regex'
```

### Section 6: Semantic Prototypes

```yaml
semantic_prototypes:
  decreto_legislativo:
    - "decreto legislativo numero del anno"
    - "d.lgs. n. del"
    - "decreto legislativo che disciplina"
    # ... 5 prototipi per tipo

  # ... 300+ prototipi totali
```

**Quando modificare**:
- Classificazione semantica imprecisa
- Aggiungere esempi di uso comune

### Section 7: Spurious Filters

```yaml
spurious_filters:
  min_length: 3  # Lunghezza minima entità

  valid_short_terms:  # Eccezioni lunghezza minima
    - "l."
    - "cc"
    - "c.c."

  spurious_words:  # Parole da ignorare
    - "il"
    - "la"
    - "del"

  spurious_patterns:  # Pattern da ignorare
    - r"^s\.\s*$"
    - r"^\d{1,2}$"  # Solo numeri 1-2 cifre

  filter_single_alpha: true  # Filtra singole lettere
  min_detection_confidence: 0.6  # Confidence minima
```

**Tuning Tips**:
- **Troppo rumore**: Aggiungere a `spurious_words` o `spurious_patterns`
- **Entità perse**: Aggiungere a `valid_short_terms`

### Section 8: Output Settings

```yaml
output_settings:
  filter_institutions: true  # Rimuove istituzioni (es. "Cassazione")
  filter_null_values: true  # Rimuove campi null dal JSON

  enable_debug_logging: false  # Log dettagliati pipeline
  log_pattern_matches: false  # Log match regex

  quality_checks:
    min_confidence_threshold: 0.6  # Min confidence output
    max_results_per_document: 100  # Max entità per documento
    deduplicate_similar: true  # Rimuove duplicati simili
    similarity_threshold: 0.85  # Soglia similarità duplicati
```

---

## Active Learning Configuration

File: `config/active_learning_config.yaml`

### Active Learning

```yaml
active_learning:
  selection_strategy: "uncertainty"  # uncertainty, random, diversity
  uncertainty_threshold: 0.7  # Min uncertainty per task creation
  default_batch_size: 10  # Documenti per batch
  use_fine_tuned_if_available: true  # Usa modello fine-tuned se disponibile

  high_confidence_threshold: 0.9  # Alta confidence (skip annotation)
  fine_tuned_confidence_threshold: 0.7  # Soglia per fine-tuned model
```

**Strategies**:
- **uncertainty**: Seleziona documenti con max incertezza (default)
- **random**: Selezione casuale
- **diversity**: Max diversità (TODO)

**Tuning**:
- **Pochi task**: Abbassare `uncertainty_threshold`
- **Troppi task**: Aumentare threshold
- **Batch size**: Aumentare per più task simultanei

### Dataset

```yaml
dataset:
  min_document_length: 20  # Lunghezza minima documento (caratteri)
  max_sequence_length: 512  # Max tokens per sequenza (BERT limit)
  min_entities_per_document: 1  # Min entità per documento valido
  require_validation: true  # Richiede validazione umana
```

### Feedback Loop

```yaml
feedback_loop:
  enabled: true  # Abilita feedback loop
  check_interval_seconds: 3600  # Check ogni ora
  auto_training_threshold: 100  # N annotazioni per auto-training
  primary_evaluation_metric: "f1_score"  # Metrica principale
```

**Auto-Training**:
- Quando raggiunte N annotazioni, training automatico
- Disabilitare settando `enabled: false`

### Labels

```yaml
labels:
  label_list:  # BIO encoding
    - O
    - B-codice_civile
    - I-codice_civile
    - B-decreto_legislativo
    - I-decreto_legislativo
    # ... tutte le label

  label2id:
    O: 0
    B-codice_civile: 1
    I-codice_civile: 2
    # ...

  id2label:
    0: O
    1: B-codice_civile
    2: I-codice_civile
    # ...
```

**Aggiungere label**:
1. Aggiungere in `label_list`: `B-{act_type}`, `I-{act_type}`
2. Aggiungere in `label2id` e `id2label`
3. Riavviare sistema

---

## Label Mapping Configuration

File: `config/label_mapping.yaml`

### Act Type → Label

```yaml
act_type_to_label:
  # Decreti
  decreto_legislativo: D.LGS
  decreto_legge: D.L
  decreto_presidente_repubblica: D.P.R
  dpcm: D.P.C.M

  # Leggi
  legge: LEGGE
  legge_costituzionale: LEGGE_COST

  # Codici
  codice_civile: CODICE_CIVILE
  codice_penale: CODICE_PENALE

  # ... mappatura completa
```

### Label Categories

```yaml
label_categories:
  Decreti:
    - D.LGS
    - D.L
    - D.P.R

  Leggi:
    - LEGGE
    - LEGGE_COST

  Codici:
    - CODICE_CIVILE
    - CODICE_PENALE

  # ... categorie complete
```

**Aggiungere mappatura**:
1. Aggiungere in `act_type_to_label`
2. Aggiungere label in categoria appropriata
3. POST `/api/v1/labels/reload` per ricaricare

---

## Database Configuration

### Connection String Format

```
postgresql://[user]:[password]@[host]:[port]/[database]

Examples:
- Local: postgresql://postgres:password@localhost:5432/legal_ner
- Remote: postgresql://user:pass@db.example.com:5432/legal_ner
- SSL: postgresql://user:pass@host:5432/db?sslmode=require
```

### Connection Pool

```python
# In code (già configurato)
engine = create_engine(
    DATABASE_URL,
    pool_size=20,        # Max connections in pool
    max_overflow=10,     # Max overflow connections
    pool_timeout=30,     # Timeout for getting connection
    pool_recycle=3600    # Recycle connections after 1h
)
```

### Migrations (Alembic)

```bash
# Genera nuova migration
alembic revision --autogenerate -m "Description"

# Applica migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1

# Check current version
alembic current

# Show history
alembic history
```

---

## Best Practices

### 1. Configuration Management

**Version Control**:
```bash
# Track configs in git
git add config/*.yaml
git commit -m "Update pipeline config"

# .env NON va in git
echo ".env" >> .gitignore
```

**Backup**:
```bash
# Backup configs prima di modifiche
cp config/pipeline_config.yaml config/pipeline_config.yaml.bak

# Con timestamp
cp config/pipeline_config.yaml "config/pipeline_config.yaml.$(date +%Y%m%d_%H%M%S)"
```

### 2. Configuration Validation

**Test dopo modifiche**:
```bash
# 1. Validare YAML sintassi
python -c "import yaml; yaml.safe_load(open('config/pipeline_config.yaml'))"

# 2. Test loading
python -c "from app.core.config_loader import get_pipeline_config; get_pipeline_config()"

# 3. Test endpoint
curl http://localhost:8000/health
```

### 3. Performance Tuning

**Profiling**:
```python
# Abilita debug logging
output_settings:
  enable_debug_logging: true
  log_pattern_matches: true

# Analizza logs
cat logs/requests/{request_id}.log | jq .
```

**Optimize per velocità**:
```yaml
# 1. Ridurre context windows
context_windows:
  entity_expansion:
    left_window: 100  # Da 150
    right_window: 100  # Da 150

# 2. Disabilitare semantic validation
confidence_thresholds:
  enable_semantic_validation_always: false

# 3. Aumentare thresholds
  minimum_detection_confidence: 0.7  # Da 0.5
```

**Optimize per accuratezza**:
```yaml
# 1. Aumentare context windows
context_windows:
  entity_expansion:
    left_window: 200
    right_window: 200

# 2. Abilitare semantic validation
confidence_thresholds:
  enable_semantic_validation_always: true

# 3. Abbassare thresholds
  minimum_detection_confidence: 0.4
```

### 4. Security

**API Key**:
```bash
# Generare API key sicura
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Settare in .env
API_KEY=generated-secure-key-here
```

**Database Password**:
```bash
# Usare password complessa
# Min 16 caratteri, mix uppercase/lowercase/numeri/simboli
```

**CORS**:
```bash
# Solo origini fidate
CORS_ORIGINS=https://your-domain.com,https://app.your-domain.com
```

### 5. Monitoring

**Health Check**:
```bash
# Automated health check
curl http://localhost:8000/health
# Expected: {"status": "ok"}
```

**Log Monitoring**:
```bash
# Tail logs in real-time
tail -f logs/app.log | jq .

# Filter by level
tail -f logs/app.log | jq 'select(.level == "error")'

# Count errors
grep -c '"level":"error"' logs/app.log
```

**Metrics**:
```bash
# Database stats
SELECT COUNT(*) FROM documents;
SELECT COUNT(*) FROM entities;
SELECT COUNT(*) FROM annotation_tasks WHERE status = 'pending';

# Model metrics
SELECT name, metrics FROM ml_models WHERE is_active = true;
```

---

## Configuration Examples

### Development

```yaml
# pipeline_config.yaml
output_settings:
  enable_debug_logging: true
  log_pattern_matches: true

# .env
LOG_LEVEL=DEBUG
FASTAPI_RELOAD=True
FLASK_DEBUG=True
```

### Production

```yaml
# pipeline_config.yaml
output_settings:
  enable_debug_logging: false
  log_pattern_matches: false

confidence_thresholds:
  minimum_detection_confidence: 0.6  # Più conservativo

# .env
LOG_LEVEL=INFO
FASTAPI_RELOAD=False
FLASK_DEBUG=False
FASTAPI_WORKERS=4
```

### High Accuracy

```yaml
confidence_thresholds:
  minimum_detection_confidence: 0.4
  enable_semantic_validation_always: true
  semantic_discrepancy_confidence_threshold: 0.85

context_windows:
  entity_expansion:
    left_window: 200
    right_window: 200
  classification_context: 300
```

### High Performance

```yaml
confidence_thresholds:
  minimum_detection_confidence: 0.7
  enable_semantic_validation_always: false

context_windows:
  entity_expansion:
    left_window: 100
    right_window: 100
  classification_context: 150

spurious_filters:
  min_detection_confidence: 0.7
```

---

**Versione**: 1.0.0
**Ultima Modifica**: 2025-10-01
