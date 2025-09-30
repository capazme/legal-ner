from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import json
from datetime import datetime
import logging

app = Flask(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
API_KEY = "your-super-secret-api-key"  # Must match settings.API_KEY in backend

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ui_app.log"),
        logging.StreamHandler()
    ]
)
app.logger = logging.getLogger(__name__)

# Logging middleware
@app.before_request
def before_request():
    """Log before each request with detailed info."""
    app.logger.info(
        "Incoming request: %s %s from %s | Query: %s | Form: %s",
        request.method,
        request.path,
        request.remote_addr,
        dict(request.args),
        dict(request.form) if request.form else {}
    )

@app.after_request
def after_request(response):
    """Log after each request with status and timing."""
    if response.status_code >= 400:
        app.logger.warning(
            "Request failed: %s %s | Status: %s | Remote: %s",
            request.method,
            request.path,
            response.status_code,
            request.remote_addr
        )
    else:
        app.logger.info(
            "Request successful: %s %s | Status: %s",
            request.method,
            request.path,
            response.status_code
        )
    return response


# === ROUTES ===

@app.route('/')
def index():
    """Main dashboard showing active learning status."""
    # Get system stats
    try:
        stats_response = requests.get(
            f"{API_BASE_URL}/feedback/system-stats",
            headers={"X-API-Key": API_KEY}
        )
        stats = stats_response.json()

        # Add default values for missing attributes
        if 'system_accuracy' not in stats:
            stats['system_accuracy'] = 0.0
        if 'golden_dataset_size' not in stats:
            stats['golden_dataset_size'] = 0
        if 'status' not in stats:
            stats['status'] = 'unknown'
        if 'predictor_type' not in stats:
            stats['predictor_type'] = 'unknown'
    except Exception as e:
        stats = {
            "error": str(e),
            "system_accuracy": 0.0,
            "golden_dataset_size": 0,
            "status": "error",
            "predictor_type": "unknown"
        }

    # Get pending annotation tasks
    try:
        tasks_response = requests.get(
            f"{API_BASE_URL}/annotations/tasks?status=pending",
            headers={"X-API-Key": API_KEY}
        )
        tasks_data = tasks_response.json()

        # Extract tasks array from response
        if isinstance(tasks_data, dict) and "tasks" in tasks_data:
            tasks = tasks_data["tasks"]
        else:
            tasks = []

        app.logger.info(f"Fetched {len(tasks)} pending tasks")
    except Exception as e:
        app.logger.error(f"Error fetching tasks: {str(e)}")
        tasks = []

    return render_template(
        'index.html',
        stats=stats,
        tasks=tasks,
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


@app.route('/trigger-active-learning', methods=['POST'])
def trigger_active_learning():
    """Trigger a new active learning iteration."""
    batch_size = request.form.get('batch_size', 10)

    app.logger.info("Triggering active learning iteration | Batch size: %s", batch_size)

    try:
        response = requests.post(
            f"{API_BASE_URL}/active-learning/trigger-iteration",
            json={"batch_size": int(batch_size)},
            headers={"X-API-Key": API_KEY}
        )
        result = response.json()

        app.logger.info("Active learning iteration triggered successfully | Result: %s", result)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        app.logger.error("Failed to trigger active learning | Error: %s", str(e))
        return jsonify({"status": "error", "message": str(e)})


@app.route('/annotate/<int:task_id>')
def annotate(task_id):
    """Show the annotation interface for a specific task."""
    try:
        # Get task details
        task_response = requests.get(
            f"{API_BASE_URL}/annotations/tasks/{task_id}",
            headers={"X-API-Key": API_KEY}
        )

        if task_response.status_code != 200:
            return f"Errore: Task non trovata (Status: {task_response.status_code})"

        task = task_response.json()
        document_id = task.get("document_id")

        # Get document details
        doc_response = requests.get(
            f"{API_BASE_URL}/documents/{document_id}",
            headers={"X-API-Key": API_KEY}
        )

        if doc_response.status_code != 200:
            return f"Errore: Documento non trovato (Status: {doc_response.status_code}, Document ID: {document_id})"

        document = doc_response.json()

        # Get entities for this document
        entities_response = requests.get(
            f"{API_BASE_URL}/entities?document_id={document_id}",
            headers={"X-API-Key": API_KEY}
        )

        if entities_response.status_code != 200:
            return f"Errore: Impossibile recuperare le entità (Status: {entities_response.status_code})"

        entities_data = entities_response.json()

        # Extract entities array from response
        if isinstance(entities_data, dict) and "entities" in entities_data:
            entities = entities_data["entities"]
        else:
            entities = []

        app.logger.info(f"Fetched {len(entities)} entities for document {document_id}")

        return render_template(
            'annotate.html',
            task=task,
            document=document,
            entities=entities
        )
    except Exception as e:
        return f"Errore: {str(e)}"


@app.route('/submit-annotation', methods=['POST'])
def submit_annotation():
    """Submit annotation feedback."""
    data = request.json

    try:
        # Format the feedback according to your API
        feedback = {
            "feedback_type": "correction",
            "original_entity": data.get("original_entity"),
            "corrected_entity": data.get("corrected_entity"),
            "confidence_score": data.get("confidence_score", 1.0),
            "notes": data.get("notes", "")
        }

        # Submit to API
        response = requests.post(
            f"{API_BASE_URL}/feedback/enhanced-feedback",
            json=feedback,
            headers={"X-API-Key": API_KEY}
        )

        # Update task status if needed
        if data.get("complete_task"):
            task_id = data.get("task_id")
            requests.put(
                f"{API_BASE_URL}/annotations/tasks/{task_id}",
                json={"status": "completed"},
                headers={"X-API-Key": API_KEY}
            )

        return jsonify({"status": "success", "result": response.json()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/train-model', methods=['POST'])
def train_model():
    """Trigger model training with collected feedback."""
    model_name = request.form.get('model_name', 'DeepMount00/Italian_NER_XXL_v2')

    app.logger.info("Triggering model training | Model: %s", model_name)

    try:
        response = requests.post(
            f"{API_BASE_URL}/active-learning/train-model",
            json={"model_name": model_name},
            headers={"X-API-Key": API_KEY}
        )
        result = response.json()

        app.logger.info("Model training started successfully | Model: %s | Result: %s", model_name, result)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        app.logger.error("Failed to start model training | Model: %s | Error: %s", model_name, str(e))
        return jsonify({"status": "error", "message": str(e)})


@app.route('/performance')
def model_performance():
    """Show model performance metrics."""
    try:
        # Get system stats for performance data
        stats_response = requests.get(
            f"{API_BASE_URL}/feedback/system-stats",
            headers={"X-API-Key": API_KEY}
        )
        stats = stats_response.json()

        return render_template('performance.html', stats=stats)
    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/create-task')
def create_task_page():
    """Show the create task page."""
    return render_template('create_task.html')


# === API ENDPOINTS ===

@app.route('/api/create-task', methods=['POST'])
def api_create_task():
    """API endpoint to create a new annotation task."""
    try:
        document_id = request.form.get('document_id')
        document_text = request.form.get('document_text')
        pre_process = request.form.get('pre_process') == 'true'

        app.logger.info(f"Creating task with: document_id={document_id}, text_length={len(document_text) if document_text else 0}")

        # If document_id is provided, use existing document
        if document_id and document_id.strip():
            # Check if document exists
            doc_response = requests.get(
                f"{API_BASE_URL}/documents/{document_id}",
                headers={"X-API-Key": API_KEY}
            )
            if doc_response.status_code != 200:
                app.logger.error(f"Document not found: {document_id}, status: {doc_response.status_code}")
                return jsonify({"status": "error", "message": f"Document not found: {document_id}"})
        else:
            # Create new document
            if not document_text or not document_text.strip():
                return jsonify({"status": "error", "message": "Document text is required"})

            app.logger.info(f"Sending to API: {json.dumps({'text': document_text})}")
            app.logger.info(f"API URL: {API_BASE_URL}/documents")

            try:
                doc_response = requests.post(
                    f"{API_BASE_URL}/documents",
                    json={"text": document_text},
                    headers={"X-API-Key": API_KEY},
                    timeout=10
                )

                app.logger.info(f"API response status: {doc_response.status_code}")
                app.logger.info(f"API response body: {doc_response.text}")

                if doc_response.status_code != 201:
                    error_detail = "Unknown error"
                    try:
                        error_data = doc_response.json()
                        error_detail = error_data.get('detail', str(error_data))
                    except:
                        error_detail = doc_response.text

                    app.logger.error(f"Failed to create document: {doc_response.status_code}, {error_detail}")
                    return jsonify({"status": "error", "message": f"Failed to create document: {error_detail}"})

                document_id = doc_response.json().get("id")
                app.logger.info(f"Created document with ID: {document_id}")
            except requests.exceptions.RequestException as e:
                app.logger.error(f"Request exception: {str(e)}")
                return jsonify({"status": "error", "message": f"API connection error: {str(e)}"})

        # Pre-process with model if requested
        if pre_process:
            process_response = requests.post(
                f"{API_BASE_URL}/process",
                json={"document_id": document_id},
                headers={"X-API-Key": API_KEY}
            )
            if process_response.status_code != 200:
                app.logger.warning(f"Failed to process document: {process_response.status_code}, {process_response.text}")

        # Create annotation task
        task_response = requests.post(
            f"{API_BASE_URL}/annotations/tasks",
            json={"document_id": document_id, "priority": "high"},
            headers={"X-API-Key": API_KEY}
        )

        if task_response.status_code != 201:
            app.logger.error(f"Failed to create task: {task_response.status_code}, {task_response.text}")
            return jsonify({"status": "error", "message": f"Failed to create task: {task_response.text}"})

        task_id = task_response.json().get("id")
        app.logger.info(f"Created task with ID: {task_id}")
        return jsonify({"status": "success", "task_id": task_id, "document_id": document_id})

    except Exception as e:
        app.logger.exception("Error creating task")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/upload-document', methods=['POST'])
def api_upload_document():
    """API endpoint to upload a document file and create a task."""
    try:
        if 'document_file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"})

        file = request.files['document_file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"})

        # Read file content
        if file:
            try:
                document_text = file.read().decode('utf-8')
            except UnicodeDecodeError:
                # Try with different encoding if utf-8 fails
                file.seek(0)
                document_text = file.read().decode('latin-1')

            pre_process = request.form.get('pre_process') == 'true'

            # Create document
            doc_response = requests.post(
                f"{API_BASE_URL}/documents",
                json={"text": document_text},
                headers={"X-API-Key": API_KEY}
            )

            if doc_response.status_code != 201:
                app.logger.error(f"Failed to create document: {doc_response.status_code}, {doc_response.text}")
                return jsonify({"status": "error", "message": f"Failed to create document: {doc_response.text}"})

            document_id = doc_response.json().get("id")
            app.logger.info(f"Created document with ID: {document_id}")

            # Pre-process with model if requested
            if pre_process:
                process_response = requests.post(
                    f"{API_BASE_URL}/process",
                    json={"document_id": document_id},
                    headers={"X-API-Key": API_KEY}
                )
                if process_response.status_code != 200:
                    app.logger.warning(f"Failed to process document: {process_response.status_code}, {process_response.text}")

            # Create annotation task
            task_response = requests.post(
                f"{API_BASE_URL}/annotations/tasks",
                json={"document_id": document_id, "priority": "high"},
                headers={"X-API-Key": API_KEY}
            )

            if task_response.status_code != 201:
                app.logger.error(f"Failed to create task: {task_response.status_code}, {task_response.text}")
                return jsonify({"status": "error", "message": f"Failed to create task: {task_response.text}"})

            task_id = task_response.json().get("id")
            app.logger.info(f"Created task with ID: {task_id}")
            return jsonify({"status": "success", "task_id": task_id, "document_id": document_id})

        return jsonify({"status": "error", "message": "Failed to process file"})

    except Exception as e:
        app.logger.exception("Error uploading document")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/submit-annotation', methods=['POST'])
def api_submit_annotation():
    """API endpoint to submit annotation feedback."""
    try:
        data = request.json
        entity_id = data.get("entity_id")
        is_correct = data.get("is_correct", False)
        task_id = data.get("task_id")

        app.logger.info(f"Submitting annotation feedback | Entity: {entity_id}, Correct: {is_correct}")

        # Get the original entity from backend
        entity_response = requests.get(
            f"{API_BASE_URL}/entities?document_id=",  # We'll get all and filter
            headers={"X-API-Key": API_KEY}
        )

        if entity_response.status_code != 200:
            return jsonify({"status": "error", "message": "Failed to fetch entity"})

        entities_data = entity_response.json()
        entities = entities_data.get("entities", [])

        # Find the entity
        original_entity = None
        document_id = None
        for entity in entities:
            if str(entity.get("id")) == str(entity_id):
                original_entity = entity
                document_id = entity.get("document_id")
                break

        if not original_entity or not document_id:
            return jsonify({"status": "error", "message": f"Entity {entity_id} not found"})

        # Determine feedback type
        if is_correct:
            feedback_type = "correct"
            corrected_entity = None
        else:
            feedback_type = "incorrect"
            # If user provided corrections
            if data.get("corrected_entity"):
                corrected_entity = {
                    "text": data["corrected_entity"]["text"],
                    "label": data["corrected_entity"]["label"],
                    "start_char": data["corrected_entity"]["start_char"],
                    "end_char": data["corrected_entity"]["end_char"]
                }
            else:
                corrected_entity = None

        # Format the feedback according to EnhancedFeedbackRequest schema
        feedback = {
            "document_id": str(document_id),
            "feedback_type": feedback_type,
            "original_entity": {
                "text": original_entity.get("text"),
                "label": original_entity.get("label"),
                "start_char": original_entity.get("start_char"),
                "end_char": original_entity.get("end_char")
            },
            "corrected_entity": corrected_entity,
            "confidence_score": 1.0,
            "notes": f"Feedback from task {task_id}"
        }

        app.logger.info(f"Sending feedback to API: {json.dumps(feedback)}")

        # Submit to API
        response = requests.post(
            f"{API_BASE_URL}/feedback/enhanced-feedback",
            json=feedback,
            headers={"X-API-Key": API_KEY}
        )

        if response.status_code != 200:
            error_detail = response.text
            app.logger.error(f"Failed to submit feedback: {response.status_code} - {error_detail}")
            return jsonify({"status": "error", "message": f"Failed to submit feedback: {error_detail}"})

        app.logger.info(f"Feedback submitted successfully")
        return jsonify({"status": "success", "result": response.json()})

    except Exception as e:
        app.logger.exception("Error submitting annotation")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/complete-task', methods=['POST'])
def api_complete_task():
    """API endpoint to mark a task as completed."""
    try:
        data = request.json
        task_id = data.get("task_id")

        if not task_id:
            return jsonify({"status": "error", "message": "Task ID is required"})

        # Update task status
        response = requests.put(
            f"{API_BASE_URL}/annotations/tasks/{task_id}",
            json={"status": "completed"},
            headers={"X-API-Key": API_KEY}
        )

        if response.status_code != 200:
            return jsonify({"status": "error", "message": "Failed to update task status"})

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/add-entity', methods=['POST'])
def api_add_entity():
    """API endpoint to add a new entity to a document."""
    try:
        data = request.json
        entity = data.get("entity")
        task_id = data.get("task_id")
        document_id = data.get("document_id")

        if not entity or not task_id or not document_id:
            return jsonify({"status": "error", "message": "Missing required data"})

        # Create entity
        entity_data = {
            "document_id": document_id,
            "text": entity.get("text"),
            "label": entity.get("label"),
            "start_char": entity.get("start_char"),
            "end_char": entity.get("end_char"),
            "confidence": entity.get("confidence", 1.0),
            "model": "manual"
        }

        response = requests.post(
            f"{API_BASE_URL}/entities",
            json=entity_data,
            headers={"X-API-Key": API_KEY}
        )

        if response.status_code != 201:
            return jsonify({"status": "error", "message": "Failed to create entity"})

        entity_id = response.json().get("id")
        return jsonify({"status": "success", "entity_id": entity_id})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/delete-entity', methods=['POST'])
def api_delete_entity():
    """API endpoint to delete an entity."""
    try:
        data = request.json
        entity_id = data.get("entity_id")

        if not entity_id:
            return jsonify({"status": "error", "message": "Entity ID is required"})

        app.logger.info(f"Deleting entity: {entity_id}")

        # Delete entity
        response = requests.delete(
            f"{API_BASE_URL}/entities/{entity_id}",
            headers={"X-API-Key": API_KEY}
        )

        if response.status_code != 200:
            app.logger.error(f"Failed to delete entity {entity_id}: {response.text}")
            return jsonify({"status": "error", "message": "Failed to delete entity"})

        app.logger.info(f"Entity {entity_id} deleted successfully")
        return jsonify({"status": "success"})

    except Exception as e:
        app.logger.exception("Error deleting entity")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/delete-task', methods=['POST'])
def api_delete_task():
    """API endpoint to delete an annotation task."""
    try:
        data = request.json
        task_id = data.get("task_id")

        if not task_id:
            return jsonify({"status": "error", "message": "Task ID is required"})

        app.logger.info(f"Deleting task: {task_id}")

        # Delete task
        response = requests.delete(
            f"{API_BASE_URL}/annotations/tasks/{task_id}",
            headers={"X-API-Key": API_KEY}
        )

        if response.status_code != 200:
            app.logger.error(f"Failed to delete task {task_id}: {response.text}")
            return jsonify({"status": "error", "message": "Failed to delete task"})

        app.logger.info(f"Task {task_id} deleted successfully")
        return jsonify({"status": "success"})

    except Exception as e:
        app.logger.exception("Error deleting task")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/update-task', methods=['POST'])
def api_update_task():
    """API endpoint to update a task's status."""
    try:
        data = request.json
        task_id = data.get("task_id")
        new_status = data.get("status")

        if not task_id or not new_status:
            return jsonify({"status": "error", "message": "Task ID and status are required"})

        app.logger.info(f"Updating task {task_id} to status: {new_status}")

        # Update task status
        response = requests.put(
            f"{API_BASE_URL}/annotations/tasks/{task_id}",
            json={"status": new_status},
            headers={"X-API-Key": API_KEY}
        )

        if response.status_code != 200:
            app.logger.error(f"Failed to update task {task_id}: {response.text}")
            return jsonify({"status": "error", "message": "Failed to update task"})

        app.logger.info(f"Task {task_id} updated successfully")
        return jsonify({"status": "success"})

    except Exception as e:
        app.logger.exception("Error updating task")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/batch-delete-tasks', methods=['POST'])
def api_batch_delete_tasks():
    """API endpoint to delete multiple tasks at once."""
    try:
        data = request.json
        task_ids = data.get("task_ids", [])

        if not task_ids or not isinstance(task_ids, list):
            return jsonify({"status": "error", "message": "Task IDs list is required"})

        app.logger.info(f"Batch deleting {len(task_ids)} tasks: {task_ids}")

        deleted_count = 0
        failed_ids = []

        # Delete each task
        for task_id in task_ids:
            try:
                response = requests.delete(
                    f"{API_BASE_URL}/annotations/tasks/{task_id}",
                    headers={"X-API-Key": API_KEY}
                )

                if response.status_code == 200:
                    deleted_count += 1
                    app.logger.info(f"Task {task_id} deleted successfully")
                else:
                    failed_ids.append(task_id)
                    app.logger.warning(f"Failed to delete task {task_id}: {response.status_code}")
            except Exception as e:
                failed_ids.append(task_id)
                app.logger.error(f"Error deleting task {task_id}: {str(e)}")

        if failed_ids:
            return jsonify({
                "status": "partial",
                "deleted_count": deleted_count,
                "failed_ids": failed_ids,
                "message": f"Deleted {deleted_count} tasks, {len(failed_ids)} failed"
            })

        app.logger.info(f"Batch delete completed: {deleted_count} tasks deleted")
        return jsonify({
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"{deleted_count} tasks deleted successfully"
        })

    except Exception as e:
        app.logger.exception("Error in batch delete tasks")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/batch-update-tasks', methods=['POST'])
def api_batch_update_tasks():
    """API endpoint to update status of multiple tasks at once."""
    try:
        data = request.json
        task_ids = data.get("task_ids", [])
        new_status = data.get("status")

        if not task_ids or not isinstance(task_ids, list):
            return jsonify({"status": "error", "message": "Task IDs list is required"})

        if not new_status:
            return jsonify({"status": "error", "message": "Status is required"})

        app.logger.info(f"Batch updating {len(task_ids)} tasks to status: {new_status}")

        updated_count = 0
        failed_ids = []

        # Update each task
        for task_id in task_ids:
            try:
                response = requests.put(
                    f"{API_BASE_URL}/annotations/tasks/{task_id}",
                    json={"status": new_status},
                    headers={"X-API-Key": API_KEY}
                )

                if response.status_code == 200:
                    updated_count += 1
                    app.logger.info(f"Task {task_id} updated successfully")
                else:
                    failed_ids.append(task_id)
                    app.logger.warning(f"Failed to update task {task_id}: {response.status_code}")
            except Exception as e:
                failed_ids.append(task_id)
                app.logger.error(f"Error updating task {task_id}: {str(e)}")

        if failed_ids:
            return jsonify({
                "status": "partial",
                "updated_count": updated_count,
                "failed_ids": failed_ids,
                "message": f"Updated {updated_count} tasks, {len(failed_ids)} failed"
            })

        app.logger.info(f"Batch update completed: {updated_count} tasks updated")
        return jsonify({
            "status": "success",
            "updated_count": updated_count,
            "message": f"{updated_count} tasks updated successfully"
        })

    except Exception as e:
        app.logger.exception("Error in batch update tasks")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/export-tasks', methods=['GET'])
def api_export_tasks():
    """API endpoint to export tasks to JSON format."""
    try:
        # Get optional filters
        status = request.args.get('status')
        document_id = request.args.get('document_id')

        app.logger.info(f"Exporting tasks | Status: {status}, Document: {document_id}")

        # Build query params
        params = {}
        if status:
            params['status'] = status
        if document_id:
            params['document_id'] = document_id

        # Get tasks from API
        tasks_response = requests.get(
            f"{API_BASE_URL}/annotations/tasks",
            params=params,
            headers={"X-API-Key": API_KEY}
        )

        if tasks_response.status_code != 200:
            return jsonify({"status": "error", "message": "Failed to fetch tasks"})

        tasks_data = tasks_response.json()
        tasks = tasks_data.get("tasks", [])

        # For each task, get associated document and entities
        export_data = []
        for task in tasks:
            try:
                # Get document
                doc_response = requests.get(
                    f"{API_BASE_URL}/documents/{task['document_id']}",
                    headers={"X-API-Key": API_KEY}
                )

                if doc_response.status_code == 200:
                    document = doc_response.json()
                else:
                    document = None

                # Get entities for this document
                entities_response = requests.get(
                    f"{API_BASE_URL}/entities?document_id={task['document_id']}",
                    headers={"X-API-Key": API_KEY}
                )

                if entities_response.status_code == 200:
                    entities_data = entities_response.json()
                    entities = entities_data.get("entities", [])
                else:
                    entities = []

                # Build export entry
                export_entry = {
                    "task": task,
                    "document": document,
                    "entities": entities
                }
                export_data.append(export_entry)

            except Exception as e:
                app.logger.error(f"Error exporting task {task['id']}: {str(e)}")

        app.logger.info(f"Exported {len(export_data)} tasks")

        # Return as JSON file download
        from flask import make_response
        response = make_response(json.dumps(export_data, indent=2, ensure_ascii=False))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=tasks_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        return response

    except Exception as e:
        app.logger.exception("Error exporting tasks")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/import-tasks', methods=['POST'])
def api_import_tasks():
    """API endpoint to import tasks from JSON format."""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"})

        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"})

        # Read and parse JSON
        try:
            import_data = json.loads(file.read().decode('utf-8'))
        except json.JSONDecodeError as e:
            return jsonify({"status": "error", "message": f"Invalid JSON: {str(e)}"})

        if not isinstance(import_data, list):
            return jsonify({"status": "error", "message": "JSON must be an array of tasks"})

        app.logger.info(f"Importing {len(import_data)} tasks")

        imported_count = 0
        failed_count = 0
        results = []

        for entry in import_data:
            try:
                task_data = entry.get("task", {})
                document_data = entry.get("document", {})
                entities_data = entry.get("entities", [])

                # Create document if text is provided
                if document_data and document_data.get("text"):
                    doc_response = requests.post(
                        f"{API_BASE_URL}/documents",
                        json={"text": document_data["text"]},
                        headers={"X-API-Key": API_KEY}
                    )

                    if doc_response.status_code == 201:
                        new_document_id = doc_response.json().get("id")
                        app.logger.info(f"Created document {new_document_id}")
                    else:
                        results.append({"status": "failed", "reason": "Failed to create document"})
                        failed_count += 1
                        continue
                else:
                    results.append({"status": "failed", "reason": "Missing document text"})
                    failed_count += 1
                    continue

                # Create entities for the document
                for entity in entities_data:
                    entity_payload = {
                        "document_id": new_document_id,
                        "text": entity.get("text"),
                        "label": entity.get("label"),
                        "start_char": entity.get("start_char"),
                        "end_char": entity.get("end_char"),
                        "confidence": entity.get("confidence", 1.0),
                        "model": entity.get("model", "imported")
                    }

                    requests.post(
                        f"{API_BASE_URL}/entities",
                        json=entity_payload,
                        headers={"X-API-Key": API_KEY}
                    )

                # Create task
                priority_value = task_data.get("priority", 0.5)
                try:
                    priority = float(priority_value)
                except (ValueError, TypeError):
                    priority = 0.5  # Default value if conversion fails

                task_payload = {
                    "document_id": new_document_id,
                    "priority": priority
                }

                task_response = requests.post(
                    f"{API_BASE_URL}/annotations/tasks",
                    json=task_payload,
                    headers={"X-API-Key": API_KEY}
                )

                if task_response.status_code == 201:
                    new_task_id = task_response.json().get("id")
                    app.logger.info(f"Created task {new_task_id}")

                    # Update status if needed
                    if task_data.get("status") and task_data["status"] != "pending":
                        requests.put(
                            f"{API_BASE_URL}/annotations/tasks/{new_task_id}",
                            json={"status": task_data["status"]},
                            headers={"X-API-Key": API_KEY}
                        )

                    results.append({
                        "status": "success",
                        "task_id": new_task_id,
                        "document_id": new_document_id
                    })
                    imported_count += 1
                else:
                    results.append({"status": "failed", "reason": "Failed to create task"})
                    failed_count += 1

            except Exception as e:
                app.logger.error(f"Error importing entry: {str(e)}")
                results.append({"status": "failed", "reason": str(e)})
                failed_count += 1

        app.logger.info(f"Import completed: {imported_count} success, {failed_count} failed")

        return jsonify({
            "status": "success",
            "imported_count": imported_count,
            "failed_count": failed_count,
            "results": results
        })

    except Exception as e:
        app.logger.exception("Error importing tasks")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/reapply-model', methods=['POST'])
def api_reapply_model():
    """API endpoint to re-apply the current model to a document."""
    try:
        data = request.json
        document_id = data.get("document_id")

        if not document_id:
            return jsonify({"status": "error", "message": "Document ID is required"})

        app.logger.info(f"Re-applying model to document: {document_id}")

        # First, delete all existing entities for this document
        entities_response = requests.get(
            f"{API_BASE_URL}/entities?document_id={document_id}",
            headers={"X-API-Key": API_KEY}
        )

        if entities_response.status_code == 200:
            entities_data = entities_response.json()
            entities = entities_data.get("entities", [])

            for entity in entities:
                requests.delete(
                    f"{API_BASE_URL}/entities/{entity['id']}",
                    headers={"X-API-Key": API_KEY}
                )

            app.logger.info(f"Deleted {len(entities)} existing entities")

        # Re-apply the model
        process_response = requests.post(
            f"{API_BASE_URL}/process",
            json={"document_id": document_id},
            headers={"X-API-Key": API_KEY}
        )

        if process_response.status_code != 200:
            app.logger.error(f"Failed to re-apply model: {process_response.text}")
            return jsonify({"status": "error", "message": f"Failed to re-apply model: {process_response.text}"})

        result = process_response.json()
        app.logger.info(f"Model re-applied successfully | Entities found: {result.get('entities_found', 0)}")

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        app.logger.exception("Error re-applying model")
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)