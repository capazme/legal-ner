# Architettura e Flusso Dati: Legal-NER-API

Questo documento è la "source of truth" tecnica per l'architettura del sistema Legal-NER-API. Descrive la struttura del progetto, il flusso di una richiesta e le responsabilità di ogni componente core.

---

## 1. Architettura Multi-Layer

L'applicazione segue un'architettura a strati per garantire la separazione delle responsabilità (separation of concerns), rendendo il sistema più modulare, testabile e manutenibile.

- **API Layer (`app/api`)**: Responsabile della gestione delle richieste HTTP, della validazione dei dati in ingresso e in uscita (tramite Pydantic) e dell'autenticazione. È il punto di ingresso dell'applicazione.
- **Service Layer (`app/services`)**: Contiene la logica di business principale. Orchestra le operazioni complesse, come l'esecuzione dell'ensemble di modelli e la validazione semantica.
- **Data Access Layer (`app/database`)**: Gestisce tutta l'interazione con il database (PostgreSQL). Fornisce un'astrazione per creare, leggere, aggiornare ed eliminare dati.
- **Core Components (`app/core`)**: Contiene la configurazione centrale, il sistema di dependency injection e la logica per il caricamento dei modelli di machine learning.

---

## 2. Struttura del Progetto

La struttura delle directory riflette l'architettura a strati.

```
legal-ner-api/
├── app/                  # Codice sorgente dell'applicazione FastAPI
│   ├── api/              # Definizione degli endpoint dell'API (API Layer)
│   ├── core/             # Componenti base e configurazione
│   ├── database/         # Interazione con il database (Data Access Layer)
│   ├── feedback/         # Gestione del ciclo di human-in-the-loop
│   ├── models/           # Definizione dei modelli Transformer
│   ├── pipelines/        # Pipeline di pre-processing e post-processing
│   └── services/         # Logica di business principale (Service Layer)
├── ml/                   # Codice per il training e la valutazione dei modelli
├── tests/                # Test unitari e di integrazione
├── docker/               # Dockerfile e script di entrypoint
├── k8s/                  # Manifest YAML per il deployment su Kubernetes
├── .venv/                # Ambiente virtuale Python
├── requirements.txt      # Dipendenze Python
├── ROADMAP.md            # Roadmap di sviluppo del progetto
└── STRUCTURE_AND_FLOW.md # Questo documento
```

---

## 3. Flusso di una Richiesta di Predizione (`/predict`)

Questo diagramma descrive il percorso di una richiesta `POST /api/v1/predict` attraverso il sistema.

1.  **Ricezione della Richiesta**: Un client invia una richiesta JSON contenente il testo da analizzare.

2.  **API Layer (FastAPI)**:
    - L'endpoint in `app/api/v1/endpoints/predict.py` riceve la richiesta.
    - **Validazione**: Pydantic (`schemas.NERRequest`) valida automaticamente che il corpo della richiesta contenga un campo `text` non vuoto.
    - **Dependency Injection**: FastAPI invoca `get_predictor()` da `app/core/dependencies.py` per ottenere un'istanza (in cache) del `EnsemblePredictor`, `get_legal_source_extractor()` per `LegalSourceExtractor` e `get_semantic_validator()` per `SemanticValidator` e `get_entity_merger()` per `EntityMerger`.

3.  **Service Layer (`EnsemblePredictor` e `LegalSourceExtractor`)**:
    - Il metodo `predict()` del `EnsemblePredictor` viene chiamato con il testo della richiesta.
    - **Predizione NER**: Il servizio esegue l'inferenza sui modelli ML configurati, estraendo le entità nominate.
    - **Active Learning**: Viene calcolata l'incertezza della predizione e il flag `requires_review` viene impostato se supera una soglia.
    - Il metodo `extract_sources()` del `LegalSourceExtractor` viene chiamato con il testo originale per identificare e strutturare le fonti giuridiche.
    - **Unione di Entità**: Il `EntityMerger` fonde entità sovrapposte o duplicate.
    - **Consenso Semantico**: Il `_semantic_consensus` raggruppa e filtra le entità basandosi sulla somiglianza testuale e sulla confidenza.
    - **Validazione Semantica**: Il `SemanticValidator` verifica le entità estratte rispetto a un set di concetti legali noti.
    - **Calibrazione (Futuro)**: I punteggi di confidenza verranno aggiustati dal `ConfidenceCalibrator`.

4.  **Data Access Layer (Database)**:
    - Il documento originale e le entità predette vengono salvate nel database tramite le funzioni CRUD (`crud.create_document`, `crud.create_entities_for_document`).

5.  **Active Learning Pipeline (Futuro)**:
    - Se `requires_review` è `True`, il risultato della predizione verrà inviato all'`ActiveLearningPipeline` per la creazione di un task di annotazione.

6.  **Risposta al Client**:
    - L'API Layer riceve le entità finali, le fonti giuridiche estratte e il flag di revisione dal service layer.
    - **Formattazione Risposta**: Pydantic (`schemas.NERResponse`) formatta i dati in una risposta JSON strutturata, includendo `entities` e `legal_sources`.
    - La risposta viene inviata al client.

7.  **Task in Background**:
    - Operazioni non bloccanti, come il logging di metriche di performance, possono essere eseguite in background (`BackgroundTasks`) per non impattare la latenza della richiesta.

---

## 4. Componenti Core

- **`EnsemblePredictor`**: Il cuore della logica di predizione. Orchestra l'esecuzione di più modelli, fonde i risultati e ne calibra l'affidabilità. È progettato per essere stateless e facilmente scalabile. Attualmente implementa la predizione a singolo modello, il calcolo dell'incertezza, il consenso semantico di base e l'integrazione con `EntityMerger` e `SemanticValidator`.

- **`LegalSourceExtractor`**: Servizio responsabile dell'estrazione di fonti giuridiche strutturate dal testo, utilizzando pattern (es. regex) per identificare elementi come tipo di atto, numero, data, ecc.

- **`SemanticValidator`**: Agisce come un "controllore di buon senso". Utilizza un set di concetti legali noti per validare che un'entità estratta (es. "D.Lgs. 23/2015") corrisponda semanticamente al suo tipo (es. `NORMATIVA`).

- **`EntityMerger`**: Servizio responsabile della fusione di entità sovrapposte o duplicate, basandosi su criteri come la sovrapposizione testuale e la corrispondenza delle etichette.

- **`ActiveLearningPipeline` (Futuro)**: Implementa la strategia di selezione dei campioni. Il suo scopo è identificare i dati più "interessanti" (quelli su cui i modelli sono più incerti o in disaccordo) da inviare agli annotatori umani, massimizzando l'efficacia del loro lavoro.

- **`DatasetBuilder` (Futuro)**: Componente offline che viene eseguito periodicamente. Raccoglie le annotazioni umane verificate, le converte nel formato corretto (es. IOB2) e crea una nuova versione del dataset di addestramento, garantendo riproducibilità e tracciabilità.

---

## 5. Schema del Database

Lo schema del database è progettato per supportare sia le operazioni dell'API sia il ciclo di vita del machine learning.

- **`documents`**: Tabella che contiene il testo grezzo originale, fungendo da sorgente di verità.
- **`entities`**: Memorizza ogni singola entità estratta da un modello, con riferimento al documento, tipo, confidenza e modello di origine.
- **`annotations`**: Tabella centrale per il feedback loop. Registra le azioni degli annotatori (conferma, rifiuto, modifica di un'entità suggerita), collegandole a un utente e a un'entità.
- **`dataset_versions`**: Traccia le versioni dei dataset di training creati, permettendo di associare ogni modello alla versione del dataset su cui è stato addestrato.
