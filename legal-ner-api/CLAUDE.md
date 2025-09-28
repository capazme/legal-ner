# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Legal-NER-API is a FastAPI-based system for Named Entity Recognition in Italian legal texts. It uses an ensemble of transformer models to extract legal entities with confidence calibration and human-in-the-loop feedback.

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

The project is in **Phase 2-3** according to ROADMAP.md:
- ✅ Basic API structure and database models
- ✅ Real model integration (two transformer models)
- ✅ Ensemble prediction with semantic consensus
- ✅ Legal source extraction and semantic validation
- ✅ Basic feedback endpoint with API key authentication
- ✅ Active learning pipeline with uncertainty calculation
- 🔄 Human-in-the-loop and dataset building in progress

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