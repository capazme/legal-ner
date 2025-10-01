# Legal-NER API Reference

> **Complete API Documentation**: All endpoints, schemas, and examples for the Legal-NER REST API.

## Base Information

- **Base URL**: `http://localhost:8000/api/v1`
- **Authentication**: API Key via header `X-API-Key`
- **Content-Type**: `application/json`
- **Response Format**: JSON

## Quick Start

```bash
# Set API key
export API_KEY="your-api-key-here"

# Test prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"text": "Secondo art. 2043 c.c., chiunque cagiona danno deve risarcirlo."}'
```

---

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Extract legal entities from text |
| `/documents` | GET, POST | Manage documents |
| `/documents/{id}` | GET, DELETE | Get/delete specific document |
| `/annotations/tasks` | GET | List annotation tasks |
| `/annotations/tasks/{id}` | GET | Get specific task |
| `/annotations/submit` | POST | Submit annotations |
| `/feedback` | POST | Submit user feedback |
| `/feedback/stats` | GET | Get feedback statistics |
| `/active-learning/start` | POST | Start active learning |
| `/active-learning/status` | GET | Get AL status |
| `/active-learning/train` | POST | Train model from feedback |
| `/labels` | GET | List all labels |
| `/labels/categories` | GET | Get labels by category |
| `/labels/mapping` | GET | Get act_type → label mapping |
| `/labels/reload` | POST | Reload label configuration |
| `/models` | GET | List ML models |
| `/models/{id}` | GET | Get model details |
| `/models/activate/{id}` | POST | Activate model |
| `/models/active` | GET | Get active model |
| `/admin/reprocess-tasks` | POST | Reprocess annotation tasks |
| `/admin/stats` | GET | Get system statistics |
| `/export/dataset` | GET | Export training dataset |
| `/export/annotations` | GET | Export annotations |
| `/health` | GET | Health check |

---

## NER & Prediction

### POST /predict

Extract legal entities from Italian legal text.

**Request:**
```json
{
  "text": "string"
}
```

**Response:**
```json
{
  "entities": [
    {
      "text": "art. 2043 c.c.",
      "label": "CODICE_CIVILE",
      "start_char": 12,
      "end_char": 26,
      "confidence": 0.99,
      "model": "specialized_pipeline",
      "stage": "structure_building",
      "structured_data": {
        "source_type": "codice_civile",
        "act_type": "codice_civile",
        "article": "2043",
        "date": null,
        "act_number": null
      }
    }
  ],
  "legal_sources": [
    {
      "source_type": "codice_civile",
      "text": "art. 2043 c.c.",
      "confidence": 0.99,
      "start_char": 12,
      "end_char": 26,
      "act_type": "codice_civile",
      "article": "2043",
      "date": null,
      "act_number": null
    }
  ]
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "text": "Ai sensi del d.lgs. 231/2001, art. 5, comma 1, le società devono adottare modelli organizzativi."
  }'
```

**Status Codes:**
- `200 OK`: Success
- `400 Bad Request`: Invalid request body
- `401 Unauthorized`: Invalid or missing API key
- `500 Internal Server Error`: Server error

---

## Documents

### GET /documents

List all documents with pagination.

**Query Parameters:**
- `skip` (int, optional): Number of records to skip (default: 0)
- `limit` (int, optional): Max records to return (default: 100)

**Response:**
```json
{
  "documents": [
    {
      "id": 1,
      "text": "Document text...",
      "source": "Optional source",
      "created_at": "2025-01-01T12:00:00",
      "updated_at": "2025-01-01T12:00:00",
      "entities_count": 5
    }
  ],
  "total": 100,
  "skip": 0,
  "limit": 100
}
```

### POST /documents

Create a new document.

**Request:**
```json
{
  "text": "Document text content",
  "source": "Optional source identifier"
}
```

**Response:**
```json
{
  "id": 123,
  "text": "Document text content",
  "source": "Optional source identifier",
  "created_at": "2025-01-01T12:00:00",
  "updated_at": "2025-01-01T12:00:00"
}
```

### GET /documents/{document_id}

Get specific document with all entities.

**Response:**
```json
{
  "id": 123,
  "text": "Document text...",
  "source": null,
  "created_at": "2025-01-01T12:00:00",
  "entities": [
    {
      "id": 456,
      "text": "art. 2043 c.c.",
      "label": "CODICE_CIVILE",
      "start_char": 12,
      "end_char": 26,
      "confidence": 0.99,
      "model": "specialized_pipeline"
    }
  ]
}
```

### DELETE /documents/{document_id}

Delete document and all associated entities (CASCADE).

**Response:**
```json
{
  "message": "Document deleted",
  "document_id": 123
}
```

---

## Annotations

### GET /annotations/tasks

List annotation tasks with filters.

**Query Parameters:**
- `status` (str, optional): Filter by status (`pending`, `in_progress`, `completed`)
- `limit` (int, optional): Max records (default: 50)

**Response:**
```json
{
  "tasks": [
    {
      "id": 1,
      "document_id": 123,
      "status": "pending",
      "priority": 0.85,
      "assigned_to": null,
      "created_at": "2025-01-01T12:00:00",
      "completed_at": null,
      "document_text": "Document preview...",
      "entities_count": 5
    }
  ]
}
```

### GET /annotations/tasks/{task_id}

Get specific annotation task with full details.

**Response:**
```json
{
  "id": 1,
  "document_id": 123,
  "status": "pending",
  "priority": 0.85,
  "document": {
    "id": 123,
    "text": "Full document text...",
    "source": null
  },
  "entities": [
    {
      "id": 456,
      "text": "art. 2043 c.c.",
      "label": "CODICE_CIVILE",
      "start_char": 12,
      "end_char": 26,
      "confidence": 0.99,
      "annotations": []
    }
  ]
}
```

### POST /annotations/submit

Submit annotations for a task.

**Request:**
```json
{
  "task_id": 1,
  "user_id": "user@example.com",
  "annotations": [
    {
      "entity_id": 456,
      "is_correct": true,
      "corrected_label": null,
      "feedback_text": null
    },
    {
      "entity_id": 457,
      "is_correct": false,
      "corrected_label": "DECRETO_LEGISLATIVO",
      "feedback_text": "Should be D.LGS not LEGGE"
    }
  ]
}
```

**Response:**
```json
{
  "message": "Annotations submitted",
  "task_id": 1,
  "annotations_count": 2
}
```

---

## Feedback

### POST /feedback

Submit user feedback on an entity.

**Request:**
```json
{
  "entity_id": 456,
  "user_id": "user@example.com",
  "is_correct": false,
  "corrected_label": "DECRETO_LEGISLATIVO",
  "feedback_text": "Classification error"
}
```

**Response:**
```json
{
  "message": "Feedback received",
  "annotation_id": 789
}
```

### GET /feedback/stats

Get feedback statistics.

**Query Parameters:**
- `user_id` (str, optional): Filter by user
- `label` (str, optional): Filter by label

**Response:**
```json
{
  "total_annotations": 1000,
  "correct_count": 850,
  "incorrect_count": 150,
  "accuracy": 0.85,
  "by_label": {
    "CODICE_CIVILE": {
      "total": 200,
      "correct": 195,
      "incorrect": 5,
      "accuracy": 0.975
    },
    "DECRETO_LEGISLATIVO": {
      "total": 150,
      "correct": 140,
      "incorrect": 10,
      "accuracy": 0.933
    }
  },
  "by_user": {
    "user@example.com": {
      "total": 500,
      "correct": 425,
      "incorrect": 75
    }
  }
}
```

---

## Active Learning

### POST /active-learning/start

Start active learning cycle to create annotation tasks.

**Request:**
```json
{
  "batch_size": 10,
  "strategy": "uncertainty"
}
```

**Parameters:**
- `batch_size` (int): Number of documents to select (default: 10)
- `strategy` (str): Selection strategy (`uncertainty`, `random`, `diversity`)

**Response:**
```json
{
  "status": "success",
  "tasks_created": 10,
  "documents_processed": 50,
  "average_uncertainty": 0.75,
  "task_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
```

### GET /active-learning/status

Get active learning system status.

**Response:**
```json
{
  "enabled": true,
  "pending_tasks": 25,
  "completed_tasks": 100,
  "total_annotations": 500,
  "auto_training_threshold": 100,
  "annotations_until_training": 0,
  "last_training": "2025-01-01T12:00:00",
  "active_model": {
    "id": 5,
    "name": "fine_tuned_20250101_120000",
    "version": "1.0.0"
  }
}
```

### POST /active-learning/train

Train new model from collected feedback.

**Request:**
```json
{
  "user_id": null,
  "training_args": {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "learning_rate": 2e-5
  }
}
```

**Response:**
```json
{
  "status": "success",
  "model_id": 6,
  "model_name": "fine_tuned_20250101_140000",
  "training_time_seconds": 3600,
  "metrics": {
    "accuracy": 0.92,
    "precision": 0.91,
    "recall": 0.93,
    "f1_score": 0.92
  },
  "dataset_info": {
    "total_samples": 500,
    "train_samples": 400,
    "val_samples": 50,
    "test_samples": 50
  }
}
```

---

## Labels

### GET /labels

List all standardized labels.

**Response:**
```json
[
  "CEDU",
  "CIRCOLARE",
  "CODICE",
  "CODICE_BENI_CULTURALI",
  "CODICE_CIVILE",
  "CODICE_CRISI_IMPRESA",
  "CODICE_PENALE",
  "CODICE_PROCEDURA_CIVILE",
  "CODICE_PROCEDURA_PENALE",
  "COSTITUZIONE",
  "D.A.R",
  "D.D",
  "D.L",
  "D.LGS",
  "D.M",
  "D.P.C.M",
  "D.P.R",
  "DEC_UE",
  "DIR_UE",
  "ISTITUZIONE",
  "L.P",
  "L.R",
  "LEGGE",
  "LEGGE_COST",
  "LEGGE_FALLIMENTARE",
  "RAC_UE",
  "REG_UE",
  "T.U",
  "T.U.B",
  "T.U.E.L",
  "T.U.F",
  "T.U.L.P.S",
  "T.U.P.S",
  "TFUE",
  "TRATTATO",
  "TRATTATO_UE",
  "UNKNOWN"
]
```

### GET /labels/categories

Get labels organized by category.

**Response:**
```json
{
  "Decreti": [
    "D.LGS",
    "D.L",
    "D.P.R",
    "D.M",
    "D.P.C.M",
    "D.D",
    "D.A.R"
  ],
  "Leggi": [
    "LEGGE",
    "LEGGE_COST",
    "L.R",
    "L.P",
    "LEGGE_FALLIMENTARE"
  ],
  "Codici": [
    "CODICE_CIVILE",
    "CODICE_PENALE",
    "CODICE_PROCEDURA_CIVILE",
    "CODICE_PROCEDURA_PENALE",
    "CODICE_CRISI_IMPRESA",
    "CODICE_BENI_CULTURALI",
    "CODICE"
  ],
  "Testi Unici": [
    "T.U",
    "T.U.B",
    "T.U.E.L",
    "T.U.F",
    "T.U.L.P.S",
    "T.U.P.S"
  ],
  "Normativa UE": [
    "DIR_UE",
    "REG_UE",
    "DEC_UE",
    "RAC_UE",
    "TRATTATO_UE",
    "TFUE"
  ],
  "Trattati": [
    "TRATTATO",
    "CONVENTION",
    "CEDU"
  ],
  "Costituzione": [
    "COSTITUZIONE"
  ],
  "Altro": [
    "CIRCOLARE",
    "ISTITUZIONE",
    "UNKNOWN"
  ]
}
```

### GET /labels/mapping

Get complete act_type → label mapping.

**Response:**
```json
{
  "decreto_legislativo": "D.LGS",
  "decreto_legge": "D.L",
  "decreto_presidente_repubblica": "D.P.R",
  "codice_civile": "CODICE_CIVILE",
  "legge": "LEGGE",
  "...": "..."
}
```

### POST /labels/reload

Reload label configuration from file.

**Response:**
```json
{
  "status": "success",
  "message": "Label configuration reloaded"
}
```

---

## Models

### GET /models

List all ML models.

**Response:**
```json
{
  "models": [
    {
      "id": 1,
      "name": "fine_tuned_20250101_120000",
      "version": "1.0.0",
      "model_path": "models/fine_tuned_20250101_120000",
      "is_active": true,
      "metrics": {
        "accuracy": 0.92,
        "precision": 0.91,
        "recall": 0.93,
        "f1_score": 0.92
      },
      "created_at": "2025-01-01T12:00:00",
      "trained_at": "2025-01-01T11:30:00"
    }
  ]
}
```

### GET /models/{model_id}

Get specific model details.

**Response:**
```json
{
  "id": 1,
  "name": "fine_tuned_20250101_120000",
  "version": "1.0.0",
  "model_path": "models/fine_tuned_20250101_120000",
  "is_active": true,
  "metrics": {
    "accuracy": 0.92,
    "precision": 0.91,
    "recall": 0.93,
    "f1_score": 0.92,
    "confusion_matrix": [[100, 5], [3, 92]]
  },
  "training_config": {
    "base_model": "dlicari/Italian-Legal-BERT",
    "num_epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5
  },
  "created_at": "2025-01-01T12:00:00",
  "trained_at": "2025-01-01T11:30:00"
}
```

### POST /models/activate/{model_id}

Activate a specific model.

**Response:**
```json
{
  "message": "Model activated",
  "model_id": 1,
  "model_name": "fine_tuned_20250101_120000"
}
```

### GET /models/active

Get currently active model.

**Response:**
```json
{
  "id": 1,
  "name": "fine_tuned_20250101_120000",
  "version": "1.0.0",
  "is_active": true,
  "metrics": {
    "accuracy": 0.92,
    "f1_score": 0.92
  }
}
```

---

## Admin

### POST /admin/reprocess-tasks

Reprocess annotation tasks with current pipeline.

**Request:**
```json
{
  "task_ids": [1, 2, 3, 4, 5],
  "replace_existing": true
}
```

**Parameters:**
- `task_ids` (list[int]): Task IDs to reprocess
- `replace_existing` (bool): Replace existing entities (default: false)

**Response:**
```json
{
  "status": "success",
  "processed": 5,
  "failed": 0,
  "results": [
    {
      "task_id": 1,
      "document_id": 123,
      "entities_found": 5,
      "entities_saved": 5,
      "status": "success"
    }
  ]
}
```

### GET /admin/stats

Get system statistics.

**Response:**
```json
{
  "documents": {
    "total": 1000,
    "with_entities": 950,
    "without_entities": 50
  },
  "entities": {
    "total": 5000,
    "by_label": {
      "CODICE_CIVILE": 800,
      "DECRETO_LEGISLATIVO": 600,
      "LEGGE": 500
    },
    "avg_per_document": 5.0,
    "avg_confidence": 0.85
  },
  "annotations": {
    "total": 2000,
    "correct": 1700,
    "incorrect": 300,
    "accuracy": 0.85
  },
  "tasks": {
    "total": 100,
    "pending": 25,
    "in_progress": 5,
    "completed": 70
  },
  "models": {
    "total": 5,
    "active": 1
  }
}
```

---

## Export

### GET /export/dataset

Export training dataset in IOB/CoNLL format.

**Query Parameters:**
- `format` (str): Export format (`iob`, `json`, `conll`) (default: `iob`)
- `include_unannotated` (bool): Include documents without annotations (default: false)

**Response:**
```json
{
  "dataset_url": "https://minio.example.com/legal-ner-datasets/export_20250101.jsonl",
  "format": "iob",
  "total_documents": 500,
  "total_samples": 2000,
  "created_at": "2025-01-01T12:00:00"
}
```

**IOB Format Example:**
```json
{
  "tokens": ["Secondo", "l'art.", "2043", "c.c.", ",", "chiunque", "..."],
  "tags": ["O", "B-CODICE_CIVILE", "I-CODICE_CIVILE", "I-CODICE_CIVILE", "O", "O", "..."]
}
```

### GET /export/annotations

Export all annotations.

**Query Parameters:**
- `user_id` (str, optional): Filter by user
- `label` (str, optional): Filter by label
- `is_correct` (bool, optional): Filter by correctness

**Response:**
```json
{
  "annotations": [
    {
      "id": 1,
      "entity_id": 456,
      "entity_text": "art. 2043 c.c.",
      "predicted_label": "CODICE_CIVILE",
      "is_correct": true,
      "corrected_label": null,
      "user_id": "user@example.com",
      "feedback_text": null,
      "created_at": "2025-01-01T12:00:00"
    }
  ],
  "total": 2000
}
```

---

## Health & Monitoring

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

**Status Codes:**
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is down

---

## Error Responses

All endpoints may return these error responses:

### 400 Bad Request
```json
{
  "detail": "Validation error message"
}
```

### 401 Unauthorized
```json
{
  "detail": "Invalid or missing API key"
}
```

### 404 Not Found
```json
{
  "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
  "detail": "Internal server error message"
}
```

---

## Rate Limiting

**Current**: No rate limiting

**Future** (planned):
- 100 requests/minute per API key
- 1000 requests/hour per API key
- Burst: 10 requests/second

---

## Versioning

**Current Version**: v1

API versioning via URL path: `/api/v1/*`

Breaking changes will increment version: `/api/v2/*`

---

## SDKs & Client Libraries

**Coming Soon**:
- Python SDK
- JavaScript SDK
- TypeScript definitions

**Current**: Use HTTP client libraries (requests, axios, fetch, etc.)

---

## Support

**Issues**: https://github.com/your-org/legal-ner/issues
**Documentation**: https://docs.legal-ner.example.com
**Contact**: support@legal-ner.example.com

---

**Version**: 1.0.0
**Last Updated**: 2025-10-01
