# legal-ner

Named-entity recognition service for Italian legal text, exposed as a FastAPI application.

Extracts legal entities (normative references, courts, parties, dates) from Italian legal documents. Part of the tooling behind [VisuaLex](https://github.com/capazme/VisuaLexAPI) and the [ALIS](https://github.com/capazme/ALIS_CORE) ecosystem.

**Status: experimental** — APIs and models may change without notice.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [API reference](docs/API_REFERENCE.md)
- [Configuration](docs/CONFIGURATION.md)

## Quick start

```bash
cd legal-ner-api
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Author

Guglielmo Puzio ([@capazme](https://github.com/capazme)) — [capazme.github.io](https://capazme.github.io)
