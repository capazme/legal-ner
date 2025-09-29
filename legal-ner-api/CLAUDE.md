# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**⚠️ UPDATED POST-RESET (Sept 28, 2025)**

## Project Overview

Legal-NER-API is a FastAPI-based system for Named Entity Recognition in Italian legal texts.

**CURRENT STATUS**: The system has undergone a complete reset of business logic while maintaining the core architecture. All ML/NER services are currently in placeholder mode pending redesign.

**ARCHITECTURE STATUS**:
- ✅ API Layer: Fully operational
- ✅ Database Layer: Fully operational
- ✅ Core Infrastructure: Fully operational
- 🔄 Service Layer: Reset to placeholders (returns empty results)

## Development Setup

### Virtual Environment
The project uses a Python virtual environment located at `.venv/`. Always activate it before running commands:
```bash
source .venv/bin/activate
```

### Running the Application
Start the development server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Database Setup
Initialize or recreate the database:
```bash
python create_db.py          # Initial setup
python recreate_db.py        # Reset database
```

## Architecture

The application follows a multi-layer architecture:

- **API Layer** (`app/api/`): FastAPI endpoints with Pydantic validation
- **Service Layer** (`app/services/`): Business logic including `EnsemblePredictor`, `LegalSourceExtractor`, `SemanticValidator`, `EntityMerger`
- **Data Access Layer** (`app/database/`): PostgreSQL interaction via SQLAlchemy
- **Core Components** (`app/core/`): Configuration, dependency injection, model loading

## Key Components

### Core Services
- **EnsemblePredictor**: Orchestrates multiple transformer models for NER
- **LegalSourceExtractor**: Extracts structured legal sources using patterns
- **SemanticValidator**: Validates entities against known legal concepts
- **EntityMerger**: Merges overlapping/duplicate entities
- **ConfidenceCalibrator**: Adjusts confidence scores (basic implementation)

### API Endpoints
- `POST /api/v1/predict`: Main NER prediction endpoint
- `POST /api/v1/feedback`: Human feedback for active learning
- `GET /health`: Health check

### Database Models
- `documents`: Original text storage
- `entities`: Extracted entities with confidence scores
- `annotations`: Human feedback for HITL training
- `dataset_versions`: Training dataset versioning

## Current Development Status

**POST-RESET STATUS**:
- ✅ API infrastructure and database models (fully operational)
- ✅ Request/response pipeline (functional with empty results)
- ✅ Database operations and CRUD (fully operational)
- ✅ Feedback system and dataset building (operational)
- ✅ Authentication and logging (operational)
- 🔄 **ALL NER/ML SERVICES RESET TO PLACEHOLDERS**
- 🚀 **Ready for complete redesign of business logic**

**PLACEHOLDER SERVICES** (require redesign):
- EnsemblePredictor → returns empty entities
- LegalSourceExtractor → returns empty sources
- SemanticValidator → pass-through validation
- ConfidenceCalibrator → pass-through calibration
- EntityMerger → pass-through merging

## Dependencies

Main dependencies include:
- FastAPI + Uvicorn for web framework
- Transformers + PyTorch for ML models
- SQLAlchemy + PostgreSQL for database
- Pydantic for data validation
- Structlog for structured logging

## Development Notes

- The project uses Italian legal domain-specific models
- Two transformer models are integrated: `dlicari/distil-ita-legal-bert` and `DeepMount00/Italian_NER_XXL_v2`
- Structured logging is configured for production readiness
- Authentication uses API keys for feedback endpoints
- Database schema supports both API operations and ML lifecycle