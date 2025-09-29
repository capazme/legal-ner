
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'legal-ner-api'))

from app.core.config import settings
from app.database.models import Entity

DATABASE_URL = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_SERVER}/{settings.POSTGRES_DB}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

db = SessionLocal()
try:
    latest_entity = db.query(Entity).order_by(Entity.id.desc()).first()
    if latest_entity:
        print(latest_entity.id)
    else:
        print("No entities found.")
finally:
    db.close()
