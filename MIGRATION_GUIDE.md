# Legal-NER System - Migration Guide

## 🎯 Obiettivo del Refactoring

Questa migrazione risolve **10 fallacie critiche** identificate nel sistema Legal-NER, trasformandolo da un sistema frammentato con dati fittizi a un **sistema production-ready** con dati reali dal database.

---

## ✅ MODIFICHE COMPLETATE

### 1. **FeedbackLoop - Database-Backed** ✨
**File**: `legal-ner-api/app/services/feedback_loop.py`

**Prima**:
- Usava file JSONL per memorizzare feedback
- Confronto `entry.text == feedback.document_id` (logica errata)
- Golden dataset con placeholder `f"document_{feedback.document_id}"`
- Metriche fake hardcoded

**Dopo**:
- **Query dirette al database** PostgreSQL
- `get_golden_dataset(db)` costruisce dataset da annotazioni validate
- **Majority voting** per determinare correttezza entità
- **Metriche reali**: Precision, Recall, F1 calcolate da annotazioni
- `calculate_quality_metrics(db)` con TP/FP/FN reali
- `check_retraining_trigger(db)` basato su annotazioni recenti

**Breaking Changes**:
- ❌ Rimosso `process_feedback()` async (non più necessario)
- ❌ Rimosso gestione file JSONL
- ✅ Tutti i metodi ora richiedono `db: Session` come parametro

---

### 2. **DatasetBuilder - Indipendente e Robusto** ✨
**File**: `legal-ner-api/app/feedback/dataset_builder.py`

**Prima**:
- Dipendeva da `FeedbackLoop.golden_dataset` (dati fittizi)
- Nessuna query al database

**Dopo**:
- **Query dirette** a `Document`, `Entity`, `Annotation`
- **Majority voting** per entità validate
- **Correzioni automatiche**: se entità marcata incorretta con `corrected_label`, usa quella
- **Validazione robusta**: controlla dataset vuoti, errori IOB conversion
- **Logging dettagliato**: traccia ogni step della conversione

**Miglioramenti**:
- ✅ Gestione overlap entità in IOB conversion
- ✅ Tokenization con `offset_mapping` per accuracy
- ✅ Upload su MinIO con size tracking

---

### 3. **Dependencies - Rimossi Antipattern** ✨
**File**: `legal-ner-api/app/core/dependencies.py`

**Prima**:
- `@lru_cache` con `Depends()` (antipattern)
- Dipendenze circolari tra `FeedbackLoop` e `DatasetBuilder`

**Dopo**:
- ✅ `FeedbackLoop` e `DatasetBuilder` sono **istanze nuove** ad ogni request (lightweight)
- ✅ Nessuna dipendenza circolare
- ✅ Codice pulito e manutenibile

---

### 4. **ModelManager - A/B Testing e Auto-Selection** ✨
**File**: `legal-ner-api/app/core/model_manager.py`

**Nuove funzionalità**:
- ✅ `list_available_models(db)` - Lista tutti i modelli con metriche
- ✅ `activate_model(db, version)` - Attiva modello specifico
- ✅ `deactivate_all_models(db)` - Torna a rule-based pipeline
- ✅ `compare_models(db, v1, v2)` - Confronta metriche tra modelli
- ✅ `auto_select_best_model(db, metric)` - Selezione automatica del migliore

**Impatto**:
- 🚀 Hot-swapping tra modelli senza restart
- 🚀 A/B testing facilitato
- 🚀 Rollback automatico se modello peggiora

---

### 5. **API Endpoints - Feedback Reali** ✨
**File**: `legal-ner-api/app/api/v1/endpoints/feedback.py`

**Nuovi endpoint**:
```
GET  /api/v1/system-stats           → Statistiche sistema (DB-backed)
GET  /api/v1/feedback-statistics    → Feedback ultimi N giorni
GET  /api/v1/quality-metrics        → Precision/Recall/F1 reali
GET  /api/v1/golden-dataset         → Golden dataset da DB
GET  /api/v1/golden-dataset/for-training → Dati per training filtrati
GET  /api/v1/retraining-status      → Check se serve retraining
POST /api/v1/build-dataset          → Build dataset da annotazioni
```

**Deprecato**:
- ❌ `POST /api/v1/enhanced-feedback` → Usa UI workflow invece

---

### 6. **API Endpoints - Model Management** ✨
**File**: `legal-ner-api/app/api/v1/endpoints/models.py`

**Nuovi endpoint**:
```
GET  /api/v1/models                      → Lista modelli disponibili
POST /api/v1/models/{version}/activate   → Attiva modello
GET  /api/v1/models/active               → Modello attivo
POST /api/v1/models/deactivate-all       → Torna a rule-based
GET  /api/v1/models/compare/v1/v2        → Confronta modelli
POST /api/v1/models/auto-select-best     → Seleziona migliore
```

---

## 🔄 WORKFLOW AGGIORNATO

### **Prima** (Broken):
```
1. UI salva feedback → FeedbackLoop con placeholder
2. DatasetBuilder legge dati fittizi da FeedbackLoop
3. Training usa dati inventati
4. Modello salvato ma MAI usato
```

### **Dopo** (Production-Ready):
```
1. UI salva annotazioni → Database (Annotation table)
2. DatasetBuilder query DB → Majority voting per golden dataset
3. Training usa dati reali validati
4. ModelManager auto-seleziona migliore e lo attiva
5. Pipeline hot-swap senza restart
```

---

## 📊 METRICHE REALI

Il sistema ora traccia metriche **calcolate da annotazioni reali**:

```python
# Esempio output di /api/v1/quality-metrics
{
  "precision": 0.92,          # TP / (TP + FP)
  "recall": 0.88,             # TP / (TP + FN)
  "f1_score": 0.90,           # 2 * (P * R) / (P + R)
  "true_positives": 234,
  "false_positives": 21,
  "false_negatives": 32,
  "total_entities": 450,
  "annotated_entities": 287
}
```

---

## 🚀 PROSSIMI STEP

### **TODO - Fix Active Learning** (Priorità Alta)
Il current `ActiveLearningManager` è inefficiente:
- ❌ Rielabora documenti già processati
- ❌ Uncertainty sampling troppo semplice (solo avg confidence)
- ❌ Nessuna cache delle predizioni

**Soluzione**:
1. Cache predizioni pipeline nel DB
2. Uncertainty sampling avanzato (entropy, margin-based, BALD)
3. Query ottimizzate per evitare riprocessing

### **TODO - Fix UI Duplicati** (Priorità Media)
`annotate.html` ha codice JavaScript duplicato (linee 276-365 e 321-365)

**Soluzione**:
1. Refactor in funzioni riutilizzabili
2. Gestione errori robusta
3. Auto-save annotazioni

---

## ⚠️ BREAKING CHANGES SUMMARY

| Componente | Breaking Change | Migrazione |
|------------|----------------|------------|
| `FeedbackLoop` | Tutti i metodi richiedono `db: Session` | Passa `db` come primo parametro |
| `FeedbackLoop.process_feedback()` | ❌ Rimosso | Usa UI workflow + `Annotation` table |
| `DatasetBuilder.__init__()` | Non richiede più `feedback_loop` | Rimuovi parametro |
| API `/enhanced-feedback` | ❌ Deprecated | Usa `/api/submit-annotation` da UI |

---

## ✅ TEST DI VERIFICA

### 1. **Test Golden Dataset**
```bash
curl -X GET "http://localhost:8000/api/v1/golden-dataset" \
  -H "X-API-Key: your-super-secret-api-key"
```

**Output atteso**: Lista documenti con entità validate

### 2. **Test Quality Metrics**
```bash
curl -X GET "http://localhost:8000/api/v1/quality-metrics" \
  -H "X-API-Key: your-super-secret-api-key"
```

**Output atteso**: Precision, Recall, F1 reali (se ci sono annotazioni)

### 3. **Test Model Management**
```bash
# Lista modelli
curl -X GET "http://localhost:8000/api/v1/models" \
  -H "X-API-Key: your-super-secret-api-key"

# Auto-seleziona migliore
curl -X POST "http://localhost:8000/api/v1/models/auto-select-best?metric=f1_score" \
  -H "X-API-Key: your-super-secret-api-key"
```

---

## 📝 NOTE FINALI

### **Cosa NON è cambiato**:
- ✅ Pipeline rule-based rimane il default (molto efficace!)
- ✅ Struttura database immutata
- ✅ UI annotation workflow identico
- ✅ MinIO storage per dataset

### **Cosa è MIGLIORATO**:
- 🚀 Dati reali invece di placeholder
- 🚀 Metriche calcolate da annotazioni
- 🚀 Model management completo
- 🚀 Zero dipendenze circolari
- 🚀 Codice manutenibile e testabile

### **Filosofia**:
> "**Il fine-tuning è OPT-IN, non obbligatorio.**
> La pipeline rule-based è già molto efficace.
> Il ML aggiunge valore solo con dati sufficienti."

---

## 🎓 RISORSE

- **Configurazione**: `config/active_learning_config.yaml`
- **Database Models**: `app/database/models.py`
- **API Docs**: http://localhost:8000/docs (dopo startup)
- **Logs**: `logs/active_learning.log`

---

## 👥 SUPPORTO

Per domande o problemi:
1. Controlla i log: `tail -f logs/active_learning.log`
2. Verifica API: `http://localhost:8000/docs`
3. Test database: Assicurati che PostgreSQL sia running

---

**Data Migrazione**: 2025-09-30
**Versione Sistema**: 2.0.0 (Database-Backed)
**Status**: ✅ Production-Ready (con fix UI e Active Learning pendenti)
