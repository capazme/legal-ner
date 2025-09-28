# Roadmap del Progetto: Legal-NER-API

Questo documento delinea la roadmap di sviluppo per l'API NER Legale. L'obiettivo è passare da un'infrastruttura di base a un sistema completo, robusto e pronto per la produzione, seguendo il blueprint architetturale.

---

### Fase 1: Fondamenta dell'Applicazione e Servizi Core (Completata)

In questa fase, l'obiettivo è costruire lo scheletro dell'applicazione, configurare i servizi essenziali e garantire che il flusso di base funzioni correttamente con componenti placeholder.

- [x] **Scaffolding del Progetto**: Creazione della struttura di directory e dei file iniziali.
- [x] **Gestione delle Dipendenze**: Impostazione di un ambiente virtuale e di un file `requirements.txt`.
- [x] **Configurazione Centrale**: Implementazione di un sistema di configurazione per gestire le impostazioni dell'applicazione (`app/core/config.py`).
- [x] **Struttura API di Base**: Creazione dell'applicazione FastAPI, del router e dell'endpoint `/predict` iniziale.
- [x] **Servizi Placeholder**: Implementazione di `EnsemblePredictor` con logica fittizia.
- [x] **Modelli di Database**: Definire gli schemi del database (es. con SQLAlchemy) per `documents`, `entities`, `annotations` in `app/database/models.py`.
- [x] **Integrazione Database**: Stabilire la connessione al database e fornire sessioni tramite dependency injection.

**Obiettivo di Fase**: Avere un'API funzionante in grado di ricevere una richiesta, passarla a un servizio e restituire una risposta mock, con la configurazione e la struttura di base pronte per l'espansione.

---

### Fase 2: Integrazione dei Modelli e Logica di Predizione (In Corso)

Questa fase si concentra sull'implementazione della logica di Machine Learning, sostituendo i placeholder con modelli reali e la logica di ensemble.

- [x] **Caricamento Modelli Reali**: Implementare il caricamento di modelli Transformer da Hugging Face o percorsi locali in `EnsemblePredictor` (due modelli integrati).
- [x] **Predizione a Singolo Modello**: Sviluppare la logica per eseguire l'inferenza con un singolo modello.
- [x] **Orchestrazione Ensemble**: Eseguire le predizioni in parallelo su tutti i modelli dell'ensemble.
- [x] **Consenso Semantico**: Implementare la logica in `_semantic_consensus` per raggruppare e filtrare le entità basandosi sulla somiglianza semantica e testuale (implementato basic exact match e integrazione `EntityMerger`).
- [x] **Estrazione Fonti Giuridiche**: Implementare un servizio `LegalSourceExtractor` per estrarre fonti giuridiche strutturate dal testo.
- [x] **Validazione Semantica**: Sviluppare il `SemanticValidator` per verificare le entità estratte rispetto a un set di concetti legali noti (implementato basic keyword lookup).
- [x] **Unione di Entità**: Implementare l'`EntityMerger` per fondere entità sovrapposte o duplicate (implementato basic overlap merging).
- [ ] **Calibrazione della Confidenza**: Sviluppare il `ConfidenceCalibrator` per aggiustare i punteggi di confidenza (placeholder).

**Obiettivo di Fase**: L'endpoint `/predict` deve essere in grado di restituire predizioni reali e di alta qualità, sfruttando la potenza dell'approccio ensemble.

---

### Fase 3: Human-in-the-Loop (HITL) e Active Learning (In Corso)

L'obiettivo di questa fase è costruire il ciclo di feedback che permette al sistema di migliorare continuamente attraverso la revisione umana.

- [x] **Endpoint di Feedback**: Implementare l'endpoint `/feedback` per ricevere le annotazioni corrette.
- [ ] **Sicurezza Endpoint**: Aggiungere l'autenticazione e l'autorizzazione per l'endpoint di feedback.
- [x] **Logica di Active Learning**: Sviluppare l'`ActiveLearningPipeline` per calcolare l'incertezza e il disaccordo del modello e decidere quando una revisione è necessaria. (Implementata logica di incertezza).
- [ ] **Creazione Task di Annotazione**: Salvare i campioni che necessitano di revisione nel database.
- [x] **Raccolta e Stoccaggio Feedback**: Implementare l' `AnnotationCollector` per processare e salvare il feedback degli annotatori. (Implementata funzione CRUD).
- [ ] **Costruzione Dataset**: Sviluppare il `DatasetBuilder` per creare e versionare nuovi dataset di training basati sul feedback raccolto.

**Obiettivo di Fase**: Avere un sistema completo per l'apprendimento continuo, dove i dati più informativi vengono selezionati per la revisione umana e utilizzati per creare nuovi dataset di addestramento.

---

### Fase 4: Ottimizzazione per la Produzione

Questa fase si concentra sulla robustezza, l'osservabilità e la scalabilità del sistema.

- [x] **Logging Strutturato**: Integrare `structlog` in tutta l'applicazione per un logging in formato JSON.
- [ ] **Monitoring**: Esportare metriche chiave (latenza, numero di predizioni, errori) utilizzando `prometheus-fastapi-instrumentator`.
- [ ] **Containerizzazione**: Creare un `Dockerfile` ottimizzato (multi-stage build) per l'applicazione.
- [ ] **Orchestrazione**: Sviluppare i manifest di Kubernetes (`deployment.yaml`, `service.yaml`) per il deployment su un cluster.
- [ ] **CI/CD**: Impostare una pipeline di base (es. con GitHub Actions) per eseguire test automatici a ogni commit.
- [ ] **Test di Integrazione e Carico**: Sviluppare test che verifichino l'interazione tra i componenti e le performance sotto carico.

**Obiettivo di Fase**: Avere un'applicazione containerizzata, monitorabile e pronta per essere deployata in un ambiente di produzione scalabile.

---

### Fase 5: Addestramento e Valutazione dei Modelli

L'ultima fase si concentra sulla chiusura del ciclo MLOps, abilitando l'addestramento e la valutazione di nuovi modelli.

- [ ] **Script di Addestramento**: Creare script nella directory `ml/` per il fine-tuning dei modelli Transformer sui dataset versionati.
- [ ] **Pipeline di Valutazione**: Sviluppare script per valutare le performance dei modelli (precision, recall, F1) su un set di test.
- [ ] **Versionamento dei Modelli**: Integrare il versionamento degli artefatti dei modelli su MinIO.
- [ ] **Automazione del Re-training**: (Opzionale) Creare un workflow per avviare automaticamente il processo di fine-tuning quando un nuovo dataset raggiunge una certa dimensione.

**Obiettivo di Fase**: Avere un processo MLOps completo che permetta di migliorare i modelli in modo riproducibile e tracciabile.
