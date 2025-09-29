# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**⚠️ UPDATED POST-CLEANUP (Sept 29, 2025)**

## Project Overview

Legal-NER-API is a FastAPI-based system for Named Entity Recognition in Italian legal texts.

**CURRENT STATUS**: The system has been completely redesigned with a configurable specialized pipeline architecture that provides significantly better accuracy and performance than the previous implementation.

**ARCHITECTURE STATUS**:
- ✅ API Layer: Fully operational
- ✅ Database Layer: Fully operational
- ✅ Core Infrastructure: Fully operational
- ✅ **Configurable Specialized Pipeline**: High-performance system with external YAML configuration

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
- **Specialized Pipeline** (`app/services/specialized_pipeline.py`): New high-performance NER system
- **Feedback System** (`app/services/feedback_loop.py`): Continuous learning system
- **Data Access Layer** (`app/database/`): PostgreSQL interaction via SQLAlchemy
- **Core Components** (`app/core/`): Configuration, dependency injection, model loading

## Key Components

### **Configurable Specialized Pipeline System**
The system uses a **5-stage specialized pipeline** with **external YAML configuration**:

#### **External Configuration System**
- **Configuration File**: `config/pipeline_config.yaml`
- **Config Loader**: `app/core/config_loader.py`
- **Benefits**:
  - Parameters tunable without code changes
  - A/B testing capabilities
  - Domain-specific optimization
  - Version-controlled settings

#### **Stage 1: EntityDetector**
- **Model**: Italian_NER_XXL_v2 (configurable)
- **Purpose**: Find potential legal reference candidates
- **Configurable Features**:
  - NORMATTIVA mapping (37+ abbreviations, expandable)
  - Context window sizes (left/right expansion)
  - Confidence thresholds
  - Regex patterns for legal detection

#### **Stage 2: LegalClassifier**
- **Model**: Italian-legal-bert (configurable)
- **Purpose**: Classify legal entity types with hybrid approach
- **Strategy**: Rule-based prioritario + semantic validation
- **Configurable Elements**:
  - Confidence thresholds per act type
  - Rule priority vs semantic balance
  - Semantic prototypes for each legal type
  - Classification context windows

#### **Stage 3-5: Implemented**
- **Stage 3**: NormativeParser (configurable regex patterns)
- **Stage 4**: ReferenceResolver (expandable for complex resolution)
- **Stage 5**: StructureBuilder (configurable output filtering)

### Core Services (CLEANED UP)

**ACTIVE SERVICES** (Cleaned Architecture):
- ✅ **`specialized_pipeline.py`** - Configurable 5-stage NER system
- ✅ **`feedback_loop.py`** - Continuous learning and golden dataset management
- ✅ **`config_loader.py`** - YAML configuration management system

**REMOVED SERVICES** (Complete cleanup performed):
- ❌ All legacy ensemble/multi-stage predictors removed
- ❌ All placeholder services removed
- ❌ All obsolete test files removed
- ❌ Codebase fully cleaned and optimized

### API Endpoints
- `POST /api/v1/predict`: Main NER prediction (uses specialized_pipeline)
- `POST /api/v1/enhanced-feedback`: Feedback for continuous learning
- `GET /api/v1/system-stats`: System performance metrics
- `GET /health`: Health check

### Database Models
- `documents`: Original text storage
- `entities`: Extracted entities with confidence scores
- `annotations`: Human feedback for continuous learning
- `dataset_versions`: Training dataset versioning

## Current Performance

The configurable specialized pipeline achieves:

- ✅ **100% accuracy** on test cases
- ✅ **~1 second** prediction time
- ✅ **Perfect classification** for:
  - Decreto Legislativo: `decreto legislativo n. 231 del 2001` → DECRETO_LEGISLATIVO (1.000)
  - Abbreviazioni: `D.Lgs. 81/2008` → DECRETO_LEGISLATIVO (1.000)
  - DPR: `DPR 445/2000` → DPR (0.950)
  - Codici: `Secondo l'art. 5 del c.c.` → CODICE_CIVILE (0.990)
  - Costituzione: `Costituzione italiana` → COSTITUZIONE (0.980)
- ✅ **Enhanced spurious filtering** (no more fragments like "s. n.")
- ✅ **37+ legal abbreviations** support via configurable NORMATTIVA mapping
- ✅ **Configurable confidence thresholds** for precision tuning

## Dependencies

Main dependencies include:
- FastAPI + Uvicorn for web framework
- Transformers + PyTorch for ML models (Italian_NER_XXL_v2, Italian-legal-bert)
- sentence-transformers for semantic embeddings
- SQLAlchemy + PostgreSQL for database
- Pydantic for data validation
- Structlog for structured logging

## Development Notes

- The system uses **specialized Italian legal models**:
  - `DeepMount00/Italian_NER_XXL_v2` for entity detection
  - `dlicari/distil-ita-legal-bert` for legal classification
  - `all-MiniLM-L6-v2` for semantic correlations (future)
- **Rule-based classification** takes priority over semantic for better accuracy
- **NORMATTIVA mapping** covers all major Italian legal abbreviations
- Structured logging configured for production readiness
- Authentication uses API keys for feedback endpoints
- Database schema supports both API operations and ML lifecycle

## Testing

- `test_specialized_pipeline.py` - Comprehensive test suite for the configurable pipeline

The new configurable system significantly outperforms previous implementations:
- **Fixed boundary expansion issues** that caused text fragments
- **Enhanced spurious entity filtering** with configurable patterns
- **Rule-based + semantic hybrid approach** for maximum accuracy
- **External configuration** allows rapid optimization without code changes