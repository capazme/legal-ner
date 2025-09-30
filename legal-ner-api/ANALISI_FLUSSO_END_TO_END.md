# Analisi Flusso End-to-End: Legal-NER Active Learning System

**Data**: 2025-09-30
**Obiettivo**: Identificare fallacie logiche e architetturali che impediscono al sistema di essere state-of-the-art

---

## 1. FLUSSO ATTUALE

### Fase 1: Creazione Task
1. Utente carica documento (testo o file)
2. Sistema crea `Document` in DB
3. Se richiesto, pre-processing con `/process` → estrae entità con specialized pipeline
4. Sistema crea `AnnotationTask`
5. Utente accede a pagina di annotazione

### Fase 2: Annotazione
1. Utente visualizza entità estratte
2. Utente può:
   - Confermare/correggere entità
   - Aggiungere nuove entità
   - Eliminare entità sbagliate
   - Re-applicare il modello

### Fase 3: Feedback & Learning
1. Feedback salvato in DB
2. Active Learning seleziona sample per annotazione
3. Training triggato manualmente
4. Fine-tuning del modello con dataset annotato

### Fase 4: Export
1. Entità convertite in formato VisuaLex
2. Export per scraping

---

## 2. CRITICITÀ ARCHITETTURALI IDENTIFICATE

### 🔴 CRITICA 1: Mancanza di ciclo di feedback automatico
**Problema**: Il sistema non chiude automaticamente il ciclo di active learning
- Non c'è trigger automatico per training quando si raggiunge una soglia di annotazioni
- Non c'è valutazione automatica delle performance dopo training
- Non c'è re-deployment automatico del modello migliorato

**Impatto**: Il modello non migliora autonomamente nel tempo
**Soluzione**: Implementare feedback loop automatico con:
- Soglie configurabili (es. ogni 100 annotazioni → training)
- Valutazione automatica su golden dataset
- A/B testing tra modello vecchio e nuovo
- Auto-deployment se accuracy migliora

---

### 🔴 CRITICA 2: Nessuna strategia di uncertainty sampling
**Problema**: Active learning seleziona sample casuali o con criterio sconosciuto
- Non c'è implementazione di uncertainty-based selection
- Non ci sono metriche di incertezza calcolate sulle predizioni
- Manca diversità nel campionamento (no cluster-based selection)

**Impatto**: Annotazioni sprecate su sample non informativi
**Soluzione**: Implementare:
- Entropy-based uncertainty per entità
- Margin sampling (differenza tra top-2 classi)
- Query-by-committee con ensemble
- Diversity sampling basato su embeddings

---

### 🔴 CRITICA 3: Specialized Pipeline non aggiornabile
**Problema**: La specialized pipeline usa modelli fissi
- Stage 1 (EntityDetector) usa sempre `Italian_NER_XXL_v2` base
- Non integra i modelli fine-tunati dal active learning
- Rule-based classifier ha priorità su modello semantico

**Impatto**: Il sistema non migliora con le annotazioni
**Soluzione**:
- Sostituire modello base con modello fine-tunato quando disponibile
- Hot-swap dei modelli senza restart
- Versioning dei modelli con rollback

---

### 🟡 CRITICA 4: Mancanza di inter-annotator agreement
**Problema**: Sistema single-annotator senza validazione
- Nessun meccanismo di doppia annotazione
- Nessuna metrica di quality control (Cohen's Kappa, Fleiss' Kappa)
- Non c'è risoluzione di conflitti tra annotatori

**Impatto**: Dataset contaminato da errori/bias di singolo annotatore
**Soluzione**:
- Multi-annotator workflow
- Calcolo agreement metrics
- Adjudication interface per risolvere conflitti

---

### 🟡 CRITICA 5: VisuaLex mapping incompleto
**Problema**: Conversione NER → VisuaLex perde informazioni
- Article extraction da regex può fallire
- Non valida se act_number/date sono coerenti
- Nessun fallback per entità ambigue

**Impatto**: Perdita di entità valide, scraper riceve dati incompleti
**Soluzione**:
- LLM-based extraction per componenti mancanti
- Validazione semantica act_type + act_number + date
- Confidence threshold per export

---

### 🟡 CRITICA 6: Nessuna gestione della distribuzione dei dati
**Problema**: Non c'è class balancing o data augmentation
- Dataset può essere sbilanciato (troppe leggi, pochi decreti)
- Nessuna tecnica di oversampling/undersampling
- Non si tracciano metriche per-class

**Impatto**: Modello biased verso classi maggioritarie
**Soluzione**:
- Class balancing durante training
- SMOTE o data augmentation per classi rare
- Per-class metrics (precision/recall/F1)

---

### 🟡 CRITICA 7: Mancanza di continual learning
**Problema**: Training da zero ogni volta
- Non usa tecniche di continual learning
- Rischio di catastrophic forgetting
- Non preserva conoscenza su sample vecchi

**Impatto**: Performance degrada su sample annotati in passato
**Soluzione**:
- Elastic Weight Consolidation (EWC)
- Replay buffer con sample vecchi
- Progressive neural networks

---

### 🟢 CRITICA 8: Nessun human-in-the-loop per casi critici
**Problema**: Sistema non escalata casi difficili
- Non identifica predizioni low-confidence
- Non chiede validazione umana su entità critiche
- Non suggerisce alternative

**Impatto**: Errori silenti su casi edge
**Soluzione**:
- Confidence threshold per human review
- Suggerimenti basati su similarity con golden dataset
- Active prompting per casi ambigui

---

### 🟢 CRITICA 9: Logging insufficiente per ML Ops
**Problema**: Logging orientato a debug, non a ML metrics
- Non traccia model drift
- Non salva feature distributions
- Non monitora latency/throughput

**Impatto**: Impossibile diagnosticare problemi in produzione
**Soluzione**:
- MLflow o Weights&Biases integration
- Feature drift detection (Kolmogorov-Smirnov test)
- Performance monitoring dashboards

---

### 🟢 CRITICA 10: Export non bidirezionale
**Problema**: VisuaLex export è unidirezionale
- Non re-importa risultati da scraper
- Non valida se scraping ha avuto successo
- Nessun feedback loop da VisuaLex al NER

**Impatto**: Errori di estrazione non vengono corretti
**Soluzione**:
- Import risultati scraping
- Validazione NER vs scraper output
- Correzione automatica basata su scraping

---

## 3. PRIORITÀ DI IMPLEMENTAZIONE

### Priority 1 (Blockers per State-of-the-Art)
1. **Uncertainty Sampling** → Senza questo, active learning non funziona
2. **Modello Aggiornabile** → Pipeline deve usare modelli fine-tunati
3. **Feedback Loop Automatico** → Sistema deve migliorare autonomamente

### Priority 2 (Quality Improvements)
4. **Inter-Annotator Agreement** → Qualità dataset
5. **Class Balancing** → Performance su classi rare
6. **Continual Learning** → No catastrophic forgetting

### Priority 3 (Production Readiness)
7. **MLOps Integration** → Monitoring e debugging
8. **Human-in-the-Loop** → Validazione casi critici
9. **VisuaLex Validation** → Export più robusto
10. **Bidirectional Export** → Feedback da scraper

---

## 4. ARCHITETTURA PROPOSTA

```
┌─────────────────────────────────────────────────────────────┐
│                     USER INTERFACE (Flask)                   │
│  [Upload] [Annotate] [Review] [Train] [Deploy] [Monitor]   │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│              FASTAPI BACKEND (Orchestrator)                  │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   ACTIVE LEARNING ENGINE (Priority 1)                │   │
│  │   - Uncertainty Sampler                              │   │
│  │   - Diversity Sampler                                │   │
│  │   - Query-by-Committee                               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   MODEL MANAGER (Priority 1)                         │   │
│  │   - Model Versioning                                 │   │
│  │   - Hot-Swap Pipeline                                │   │
│  │   - A/B Testing                                      │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   SPECIALIZED PIPELINE (Updated)                     │   │
│  │   Stage 1: EntityDetector → Uses LATEST fine-tuned  │   │
│  │   Stage 2: LegalClassifier → Hybrid (semantic+rule) │   │
│  │   Stage 3-5: Parser/Resolver/Builder                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   TRAINING ORCHESTRATOR (Priority 1)                 │   │
│  │   - Threshold Monitor (auto-trigger training)        │   │
│  │   - Class Balancer                                   │   │
│  │   - Continual Learning (EWC)                         │   │
│  │   - Model Evaluator                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │   ANNOTATION QUALITY CONTROL (Priority 2)            │   │
│  │   - Inter-Annotator Agreement                        │   │
│  │   - Conflict Resolution                              │   │
│  │   - Outlier Detection                                │   │
│  └─────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                    STORAGE LAYER                             │
│  [PostgreSQL] [MinIO] [Redis Cache] [MLflow]               │
└─────────────────────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│               EXTERNAL INTEGRATIONS                          │
│  [VisuaLex Scraper] ← Bidirectional → [Feedback Import]    │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. METRICHE DA TRACCIARE

### Performance Metrics
- Per-class Precision/Recall/F1
- Overall Accuracy
- Cohen's Kappa (inter-annotator)
- Model Confidence Distribution

### Active Learning Metrics
- Uncertainty Score per Sample
- Diversity Score
- Annotation Efficiency (Δ Performance / Annotation)
- Oracle Query Rate

### Production Metrics
- Inference Latency (p50, p95, p99)
- Throughput (requests/sec)
- Model Drift (KS statistic)
- Error Rate by Entity Type

---

## 6. CONCLUSIONI

Il sistema attuale è **funzionale ma non state-of-the-art** perché:

1. ❌ **Active Learning è passivo**: Non seleziona sample informativi
2. ❌ **Pipeline non migliora**: Modelli fissi, non aggiornati
3. ❌ **Nessun controllo qualità**: Single-annotator, no validation
4. ❌ **Training manuale**: Nessun ciclo automatico
5. ⚠️ **Export limitato**: Unidirezionale, nessuna validazione

Per diventare **state-of-the-art**, servono:
- Uncertainty sampling intelligente
- Hot-swap modelli fine-tunati
- Feedback loop automatico
- Quality control annotations
- MLOps integration completa

**Stima effort**:
- Priority 1 → 2-3 settimane
- Priority 2 → 1-2 settimane
- Priority 3 → 1-2 settimane

**Total**: ~6 settimane per sistema production-ready e state-of-the-art