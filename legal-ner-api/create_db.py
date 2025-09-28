from app.database.database import Base, engine
from app.database.models import Document, Entity, Annotation, DatasetVersion

print("Creating database tables...")
Base.metadata.create_all(bind=engine)
print("Database tables created.")
