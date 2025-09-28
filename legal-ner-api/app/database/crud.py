from sqlalchemy.orm import Session
from . import models
from app.api.v1 import schemas

def create_document(db: Session, text: str) -> models.Document:
    """Creates a new document in the database."""
    db_document = models.Document(text=text)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def create_entities_for_document(db: Session, document_id: int, entities: list[schemas.Entity]):
    """Creates entity records for a given document."""
    db_entities = []
    for entity in entities:
        db_entity = models.Entity(
            document_id=document_id,
            text=entity.text,
            start_char=entity.start_char,
            end_char=entity.end_char,
            label=entity.label,
            confidence=entity.confidence,
            model=entity.model
        )
        db_entities.append(db_entity)
    
    db.add_all(db_entities)
    db.commit()
    return db_entities

def create_annotations(db: Session, annotations: list[schemas.Annotation], user_id: str):
    """Creates annotation records from user feedback."""
    db_annotations = []
    for annotation in annotations:
        db_annotation = models.Annotation(
            entity_id=annotation.entity_id,
            user_id=user_id,
            is_correct=annotation.is_correct,
            corrected_label=annotation.corrected_label
        )
        db_annotations.append(db_annotation)
    
    db.add_all(db_annotations)
    db.commit()
    return db_annotations

def create_annotation_task(db: Session, document_id: int, priority: float = 0.0) -> models.AnnotationTask:
    """Creates a new annotation task for a document."""
    db_task = models.AnnotationTask(document_id=document_id, priority=priority)
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

