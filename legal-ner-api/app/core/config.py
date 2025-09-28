from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    # API Settings
    PROJECT_NAME: str = "Legal-NER-API"
    API_V1_STR: str = "/api/v1"
    API_KEY: str = "your-super-secret-api-key"

    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "legal_ner"
    DATABASE_URI: Optional[str] = None

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # MinIO
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "ml-artifacts"

    # ML Models
    # Example: ["model1_path_or_name", "model2_path_or_name"]
    ENSEMBLE_MODELS: List[str] = ["dlicari/distil-ita-legal-bert", "DeepMount00/Italian_NER_XXL_v2"]
    UNCERTAINTY_THRESHOLD: float = 0.5

    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
