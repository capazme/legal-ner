#!/usr/bin/env python3
"""
Script per verificare lo stato del database
"""

from app.database.database import SessionLocal
from app.database import models
from sqlalchemy import func

db = SessionLocal()

print("=" * 60)
print("DATABASE STATUS CHECK")
print("=" * 60)

# Check tasks
tasks = db.query(models.AnnotationTask).all()
print(f"\n📋 AnnotationTasks: {len(tasks)}")
if tasks:
    pending = [t for t in tasks if t.status == 'pending']
    completed = [t for t in tasks if t.status == 'completed']
    print(f"   - Pending: {len(pending)}")
    print(f"   - Completed: {len(completed)}")
    print(f"\n   Last 3 tasks:")
    for t in tasks[-3:]:
        print(f"     • Task {t.id}: doc={t.document_id}, status={t.status}")

# Check entities
entities = db.query(models.Entity).all()
print(f"\n🏷️  Entities: {len(entities)}")
if entities:
    print(f"   Last 3 entities:")
    for e in entities[-3:]:
        print(f"     • Entity {e.id}: doc={e.document_id}, label={e.label}, text={e.text[:30]}...")

# Check annotations (THIS IS THE KEY)
annotations = db.query(models.Annotation).all()
print(f"\n✍️  Annotations (Feedback): {len(annotations)}")
if annotations:
    print(f"   All annotations:")
    for a in annotations:
        print(f"     • Annotation {a.id}: entity={a.entity_id}, correct={a.is_correct}, user={a.user_id}")
else:
    print("   ⚠️  NO ANNOTATIONS FOUND - This is why training fails!")

# Check documents with annotations
docs_with_annotations = (
    db.query(models.Document)
    .join(models.Entity, models.Document.id == models.Entity.document_id)
    .join(models.Annotation, models.Entity.id == models.Annotation.entity_id)
    .distinct()
    .all()
)
print(f"\n📄 Documents with annotations: {len(docs_with_annotations)}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Tasks exist: {'✅' if tasks else '❌'}")
print(f"Entities exist: {'✅' if entities else '❌'}")
print(f"Annotations exist: {'✅' if annotations else '❌'}")
print(f"Ready for training: {'✅' if docs_with_annotations else '❌'}")
print()

if not annotations and tasks:
    print("⚠️  You have tasks but NO annotations!")
    print("   This means you haven't submitted feedback yet.")
    print("   Go to the UI and click ✓ or ✗ on some entities!")
    print()

db.close()
