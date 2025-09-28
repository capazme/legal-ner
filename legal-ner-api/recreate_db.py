from app.database.database import Base, engine
from app.database.models import Document, Entity, Annotation, DatasetVersion, AnnotationTask

print("Dropping all database tables...")
Base.metadata.drop_all(bind=engine)
print("Creating all database tables...")
Base.metadata.create_all(bind=engine)
print("Database tables recreated.")
