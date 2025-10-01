# Sistema di Labeling Uniforme

## Panoramica

Il sistema di NER utilizza un approccio a due livelli per la gestione delle label:

1. **Act Types (Interni)**: Identificatori interni usati dalla specialized pipeline (es. `decreto_legislativo`, `codice_civile`)
2. **Labels Standardizzate (Display)**: Etichette visualizzate nell'UI e salvate nel database (es. `D.LGS`, `CODICE_CIVILE`)

## Mappatura Centralizzata

La mappatura tra act_types e labels è centralizzata nel modulo `app/core/label_mapping.py`.

### Struttura

```python
ACT_TYPE_TO_LABEL = {
    'decreto_legislativo': 'D.LGS',
    'codice_civile': 'CODICE_CIVILE',
    # ...
}
```

### Categorie di Label

Le label sono organizzate in categorie per facilitare la navigazione nell'UI:

- **Decreti**: D.LGS, D.L, D.P.R, D.M, D.P.C.M, D.D, D.A.R
- **Leggi**: LEGGE, LEGGE_COST, L.R, L.P, LEGGE_FALLIMENTARE
- **Codici**: CODICE_CIVILE, CODICE_PENALE, CODICE_PROCEDURA_CIVILE, CODICE_PROCEDURA_PENALE, CODICE_CRISI_IMPRESA, CODICE_BENI_CULTURALI, CODICE
- **Testi Unici**: T.U, T.U.B, T.U.E.L, T.U.F, T.U.L.P.S, T.U.P.S
- **Normativa UE**: DIR_UE, REG_UE, DEC_UE, RAC_UE, TRATTATO_UE, TFUE
- **Trattati**: TRATTATO, CONVENTION, CEDU
- **Costituzione**: COSTITUZIONE
- **Altro**: CIRCOLARE, ISTITUZIONE, UNKNOWN

## Utilizzo

### Conversione Act Type → Label

```python
from app.core.label_mapping import act_type_to_label

act_type = "decreto_legislativo"
label = act_type_to_label(act_type)  # Restituisce "D.LGS"
```

### Conversione Label → Act Type

```python
from app.core.label_mapping import label_to_act_type

label = "D.LGS"
act_type = label_to_act_type(label)  # Restituisce "decreto_legislativo"
```

### Ottenere Categoria di una Label

```python
from app.core.label_mapping import get_label_category

label = "D.LGS"
category = get_label_category(label)  # Restituisce "Decreti"
```

### Validare una Label

```python
from app.core.label_mapping import validate_label

is_valid = validate_label("D.LGS")  # Restituisce True
is_valid = validate_label("INVALID")  # Restituisce False
```

## File che Utilizzano la Mappatura

I seguenti file sono stati aggiornati per usare la mappatura centralizzata:

1. **app/api/v1/endpoints/admin.py**
   - Reprocessing delle task: converte act_type in label prima di salvare

2. **app/api/v1/endpoints/process.py**
   - Processamento documenti: converte act_type in label prima di salvare

3. **app/feedback/active_learning.py**
   - Creazione task di annotazione: converte act_type in label

4. **app/database/crud.py**
   - Creazione entità: converte valori se necessario

5. **app/api/v1/endpoints/annotations.py**
   - Creazione manuale entità: converte valori se necessario

6. **app/api/v1/endpoints/labels.py**
   - Endpoint API per ottenere le label disponibili

## API Endpoints

### GET /api/v1/labels
Restituisce la lista di tutte le label standardizzate disponibili.

**Response:**
```json
[
  "CIRCOLARE",
  "CODICE",
  "CODICE_BENI_CULTURALI",
  "CODICE_CIVILE",
  ...
]
```

### GET /api/v1/labels/categories
Restituisce le label organizzate per categoria.

**Response:**
```json
{
  "Decreti": ["D.LGS", "D.L", "D.P.R", ...],
  "Leggi": ["LEGGE", "LEGGE_COST", ...],
  ...
}
```

## Aggiungere una Nuova Label

Per aggiungere una nuova label al sistema:

1. Aggiungi la mappatura in `app/core/label_mapping.py`:
   ```python
   ACT_TYPE_TO_LABEL = {
       ...
       'nuovo_tipo_atto': 'NUOVA_LABEL',
   }
   ```

2. Aggiungi la label alla categoria appropriata:
   ```python
   LABEL_CATEGORIES = {
       'Categoria Appropriata': [..., 'NUOVA_LABEL'],
   }
   ```

3. Se necessario, aggiorna anche la configurazione della specialized pipeline in `config/pipeline_config.yaml` per riconoscere il nuovo tipo di atto.

## Database

Nel database, la colonna `entities.label` memorizza sempre la **label standardizzata** (es. `D.LGS`, `CODICE_CIVILE`), mai l'act_type interno.

Questo garantisce coerenza tra:
- Dati visualizzati nell'UI
- Dati salvati nel database
- Dati restituiti dalle API

## Note Importanti

1. **Mai usare act_type direttamente nel database**: Usare sempre `act_type_to_label()` per convertire prima di salvare
2. **Consistenza**: Tutte le parti del sistema devono usare la mappatura centralizzata
3. **Fallback**: Se un act_type non è nella mappatura, viene convertito in maiuscolo (es. `nuovo_tipo` → `NUOVO_TIPO`)
4. **Case insensitive**: Le conversioni gestiscono automaticamente maiuscole/minuscole
