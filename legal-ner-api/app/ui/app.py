from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
import requests
import json
from datetime import datetime
import logging

app = Flask(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:8000/api/v1"
API_KEY = "your-super-secret-api-key"  # Must match settings.API_KEY in backend

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ui_app.log"),
        logging.StreamHandler()
    ]
)
app.logger = logging.getLogger(__name__)


# ============================================================================
# MIDDLEWARE
# ============================================================================

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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _make_api_request(method, endpoint, **kwargs):
    """
    Make a request to the FastAPI backend with proper headers.

    Args:
        method: HTTP method (get, post, put, delete)
        endpoint: API endpoint path (will be prefixed with API_BASE_URL)
        **kwargs: Additional arguments to pass to requests method

    Returns:
        requests.Response object
    """
    headers = kwargs.pop('headers', {})
    headers['X-API-Key'] = API_KEY

    url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"

    method_map = {
        'get': requests.get,
        'post': requests.post,
        'put': requests.put,
        'delete': requests.delete
    }

    return method_map[method.lower()](url, headers=headers, **kwargs)


def _extract_response_data(response, data_key=None):
    """
    Extract data from API response with error handling.

    Args:
        response: requests.Response object
        data_key: Optional key to extract from response JSON

    Returns:
        Extracted data or None if error
    """
    if response.status_code >= 400:
        return None

    data = response.json()

    if data_key and isinstance(data, dict):
        return data.get(data_key, data)

    return data


# ============================================================================
# PAGE ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard showing active learning status."""
    # Get system stats
    try:
        stats_response = _make_api_request('get', 'system-stats')
        stats = stats_response.json()

        # Add default values for missing attributes
        stats.setdefault('system_accuracy', 0.0)
        stats.setdefault('golden_dataset_size', 0)
        stats.setdefault('status', 'unknown')
        stats.setdefault('predictor_type', 'unknown')
    except Exception as e:
        app.logger.error(f"Error fetching stats: {str(e)}")
        stats = {
            "error": str(e),
            "system_accuracy": 0.0,
            "golden_dataset_size": 0,
            "status": "error",
            "predictor_type": "unknown"
        }

    # Get annotation tasks
    try:
        tasks_response = _make_api_request('get', 'annotations/tasks')
        tasks_data = tasks_response.json()
        tasks = tasks_data.get("tasks", []) if isinstance(tasks_data, dict) else []
        app.logger.info(f"Fetched {len(tasks)} total tasks")
    except Exception as e:
        app.logger.error(f"Error fetching tasks: {str(e)}")
        tasks = []

    return render_template(
        'index.html',
        stats=stats,
        tasks=tasks,
        current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )


@app.route('/annotate/<int:task_id>')
def annotate(task_id):
    """Show the annotation interface for a specific task."""
    try:
        # Get task details
        task_response = _make_api_request('get', f'annotations/tasks/{task_id}')
        if task_response.status_code != 200:
            return f"Errore: Task non trovata (Status: {task_response.status_code})"

        task = task_response.json()
        document_id = task.get("document_id")

        # Get document details
        doc_response = _make_api_request('get', f'documents/{document_id}')
        if doc_response.status_code != 200:
            return f"Errore: Documento non trovato (Status: {doc_response.status_code}, Document ID: {document_id})"

        document = doc_response.json()

        # Get entities for this document
        entities_response = _make_api_request('get', f'entities?document_id={document_id}')
        if entities_response.status_code != 200:
            return f"Errore: Impossibile recuperare le entità (Status: {entities_response.status_code})"

        entities_data = entities_response.json()
        entities = entities_data.get("entities", []) if isinstance(entities_data, dict) else []
        app.logger.info(f"Fetched {len(entities)} entities for document {document_id}")

        # Get labels from the backend
        labels = []
        try:
            labels_response = _make_api_request('get', 'labels')
            if labels_response.status_code == 200:
                labels = labels_response.json()
        except Exception as e:
            app.logger.error(f"Error fetching labels: {str(e)}")

        return render_template(
            'annotate.html',
            task=task,
            document=document,
            entities=entities,
            labels=labels
        )
    except Exception as e:
        return f"Errore: {str(e)}"


@app.route('/performance')
def model_performance():
    """Show model performance metrics."""
    try:
        stats_response = _make_api_request('get', 'system-stats')
        stats = stats_response.json()
        return render_template('performance.html', stats=stats)
    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/create-task')
def create_task_page():
    """Show the create task page."""
    return render_template('create_task.html')


@app.route('/admin')
def admin_panel():
    """Admin panel for system management."""
    return render_template('admin.html')


# ============================================================================
# ACTIVE LEARNING ROUTES
# ============================================================================

@app.route('/trigger-active-learning', methods=['POST'])
def trigger_active_learning():
    """Trigger a new active learning iteration."""
    batch_size = request.form.get('batch_size', 10)
    app.logger.info("Triggering active learning iteration | Batch size: %s", batch_size)

    try:
        response = _make_api_request(
            'post',
            'active-learning/trigger-iteration',
            json={"batch_size": int(batch_size)}
        )

        if response.status_code != 200:
            error_detail = response.text
            app.logger.error(f"Failed to trigger active learning: {response.status_code} - {error_detail}")
            return jsonify({"status": "error", "message": f"Failed to trigger active learning: {error_detail}"})

        result = response.json()
        app.logger.info("Active learning iteration triggered successfully | Result: %s", result)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        app.logger.error("Failed to trigger active learning | Error: %s", str(e))
        return jsonify({"status": "error", "message": str(e)})


@app.route('/train-model', methods=['POST'])
def train_model():
    """Trigger model training with collected feedback."""
    model_name = request.form.get('model_name', 'DeepMount00/Italian_NER_XXL_v2')
    app.logger.info("Triggering model training | Model: %s", model_name)

    try:
        response = _make_api_request(
            'post',
            'active-learning/train-model',
            json={"model_name": model_name}
        )
        result = response.json()

        app.logger.info("Model training started successfully | Model: %s | Result: %s", model_name, result)
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        app.logger.error("Failed to start model training | Model: %s | Error: %s", model_name, str(e))
        return jsonify({"status": "error", "message": str(e)})


# ============================================================================
# LEGACY ROUTES (DEPRECATED - to be removed)
# ============================================================================

@app.route('/submit-annotation', methods=['POST'])
def submit_annotation():
    """
    DEPRECATED: Use /api/submit-annotation instead.
    Submit annotation feedback using old endpoint.
    """
    data = request.json

    try:
        feedback = {
            "feedback_type": "correction",
            "original_entity": data.get("original_entity"),
            "corrected_entity": data.get("corrected_entity"),
            "confidence_score": data.get("confidence_score", 1.0),
            "notes": data.get("notes", "")
        }

        response = _make_api_request('post', 'enhanced-feedback', json=feedback)

        # Update task status if needed
        if data.get("complete_task"):
            task_id = data.get("task_id")
            _make_api_request(
                'put',
                f'annotations/tasks/{task_id}',
                json={"status": "completed"}
            )

        return jsonify({"status": "success", "result": response.json()})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ============================================================================
# API ENDPOINTS - TASKS
# ============================================================================

@app.route('/api/create-task', methods=['POST'])
def api_create_task():
    """Create a new annotation task."""
    try:
        document_id = request.form.get('document_id')
        document_text = request.form.get('document_text')
        pre_process = request.form.get('pre_process') == 'true'

        app.logger.info(f"Creating task | document_id={document_id}, text_length={len(document_text) if document_text else 0}")

        # If document_id is provided, validate it exists
        if document_id and document_id.strip():
            doc_response = _make_api_request('get', f'documents/{document_id}')
            if doc_response.status_code != 200:
                app.logger.error(f"Document not found: {document_id}")
                return jsonify({"status": "error", "message": f"Document not found: {document_id}"})
        else:
            # Create new document
            if not document_text or not document_text.strip():
                return jsonify({"status": "error", "message": "Document text is required"})

            try:
                doc_response = _make_api_request(
                    'post',
                    'documents',
                    json={"text": document_text},
                    timeout=10
                )

                if doc_response.status_code != 201:
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
            process_response = _make_api_request('post', 'process', json={"document_id": document_id})
            if process_response.status_code != 200:
                app.logger.warning(f"Failed to process document: {process_response.status_code}, {process_response.text}")

        # Create annotation task
        task_response = _make_api_request(
            'post',
            'annotations/tasks',
            json={"document_id": document_id, "priority": 0.8}
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
    """Upload a document file and create a task."""
    try:
        if 'document_file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"})

        file = request.files['document_file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"})

        # Read file content
        try:
            document_text = file.read().decode('utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            document_text = file.read().decode('latin-1')

        pre_process = request.form.get('pre_process') == 'true'

        # Create document
        doc_response = _make_api_request('post', 'documents', json={"text": document_text})

        if doc_response.status_code != 201:
            app.logger.error(f"Failed to create document: {doc_response.status_code}, {doc_response.text}")
            return jsonify({"status": "error", "message": f"Failed to create document: {doc_response.text}"})

        document_id = doc_response.json().get("id")
        app.logger.info(f"Created document with ID: {document_id}")

        # Pre-process with model if requested
        if pre_process:
            process_response = _make_api_request('post', 'process', json={"document_id": document_id})
            if process_response.status_code != 200:
                app.logger.warning(f"Failed to process document: {process_response.status_code}, {process_response.text}")

        # Create annotation task
        task_response = _make_api_request(
            'post',
            'annotations/tasks',
            json={"document_id": document_id, "priority": 0.8}
        )

        if task_response.status_code != 201:
            app.logger.error(f"Failed to create task: {task_response.status_code}, {task_response.text}")
            return jsonify({"status": "error", "message": f"Failed to create task: {task_response.text}"})

        task_id = task_response.json().get("id")
        app.logger.info(f"Created task with ID: {task_id}")
        return jsonify({"status": "success", "task_id": task_id, "document_id": document_id})

    except Exception as e:
        app.logger.exception("Error uploading document")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/complete-task', methods=['POST'])
def api_complete_task():
    """Mark a task as completed."""
    try:
        data = request.json
        task_id = data.get("task_id")

        if not task_id:
            return jsonify({"status": "error", "message": "Task ID is required"})

        response = _make_api_request('put', f'annotations/tasks/{task_id}', json={"status": "completed"})

        if response.status_code != 200:
            return jsonify({"status": "error", "message": "Failed to update task status"})

        return jsonify({"status": "success"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/delete-task', methods=['POST'])
def api_delete_task():
    """Delete an annotation task."""
    try:
        data = request.json
        task_id = data.get("task_id")

        if not task_id:
            return jsonify({"status": "error", "message": "Task ID is required"})

        app.logger.info(f"Deleting task: {task_id}")

        response = _make_api_request('delete', f'annotations/tasks/{task_id}')

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
    """Update a task's status."""
    try:
        data = request.json
        task_id = data.get("task_id")
        new_status = data.get("status")

        if not task_id or not new_status:
            return jsonify({"status": "error", "message": "Task ID and status are required"})

        app.logger.info(f"Updating task {task_id} to status: {new_status}")

        response = _make_api_request('put', f'annotations/tasks/{task_id}', json={"status": new_status})

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
    """Delete multiple tasks at once."""
    try:
        data = request.json
        task_ids = data.get("task_ids", [])

        if not task_ids or not isinstance(task_ids, list):
            return jsonify({"status": "error", "message": "Task IDs list is required"})

        app.logger.info(f"Batch deleting {len(task_ids)} tasks: {task_ids}")

        deleted_count = 0
        failed_ids = []

        for task_id in task_ids:
            try:
                response = _make_api_request('delete', f'annotations/tasks/{task_id}')

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
    """Update status of multiple tasks at once."""
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

        for task_id in task_ids:
            try:
                response = _make_api_request('put', f'annotations/tasks/{task_id}', json={"status": new_status})

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
    """Export tasks to JSON format."""
    try:
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
        tasks_response = _make_api_request('get', 'annotations/tasks', params=params)

        if tasks_response.status_code != 200:
            return jsonify({"status": "error", "message": "Failed to fetch tasks"})

        tasks_data = tasks_response.json()
        tasks = tasks_data.get("tasks", [])

        # For each task, get associated document and entities
        export_data = []
        for task in tasks:
            try:
                # Get document
                doc_response = _make_api_request('get', f"documents/{task['document_id']}")
                document = doc_response.json() if doc_response.status_code == 200 else None

                # Get entities
                entities_response = _make_api_request('get', f"entities?document_id={task['document_id']}")
                if entities_response.status_code == 200:
                    entities_data = entities_response.json()
                    entities = entities_data.get("entities", [])
                else:
                    entities = []

                export_data.append({
                    "task": task,
                    "document": document,
                    "entities": entities
                })

            except Exception as e:
                app.logger.error(f"Error exporting task {task['id']}: {str(e)}")

        app.logger.info(f"Exported {len(export_data)} tasks")

        # Return as JSON file download
        response = make_response(json.dumps(export_data, indent=2, ensure_ascii=False))
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        response.headers['Content-Disposition'] = f'attachment; filename=tasks_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

        return response

    except Exception as e:
        app.logger.exception("Error exporting tasks")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/import-tasks', methods=['POST'])
def api_import_tasks():
    """Import tasks from JSON format."""
    try:
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

                # Create document
                if document_data and document_data.get("text"):
                    doc_response = _make_api_request('post', 'documents', json={"text": document_data["text"]})

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

                # Create entities
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

                    _make_api_request('post', 'entities', json=entity_payload)

                # Create task
                priority = float(task_data.get("priority", 0.5)) if task_data.get("priority") else 0.5

                task_response = _make_api_request(
                    'post',
                    'annotations/tasks',
                    json={"document_id": new_document_id, "priority": priority}
                )

                if task_response.status_code == 201:
                    new_task_id = task_response.json().get("id")
                    app.logger.info(f"Created task {new_task_id}")

                    # Update status if needed
                    if task_data.get("status") and task_data["status"] != "pending":
                        _make_api_request(
                            'put',
                            f'annotations/tasks/{new_task_id}',
                            json={"status": task_data["status"]}
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


# ============================================================================
# API ENDPOINTS - ENTITIES
# ============================================================================

@app.route('/api/create-entity', methods=['POST'])
def api_create_entity():
    """Create a new entity from manual annotation (text selection)."""
    try:
        data = request.json
        document_id = data.get("document_id")
        text = data.get("text")
        label = data.get("label")
        start_char = data.get("start_char")
        end_char = data.get("end_char")
        confidence = data.get("confidence", 1.0)
        model = data.get("model", "manual")

        if not all([document_id, text, label, start_char is not None, end_char is not None]):
            return jsonify({"status": "error", "message": "Missing required fields"})

        app.logger.info(f"Creating entity | Document: {document_id}, Label: {label}, Text: {text[:30]}...")

        entity_data = {
            "document_id": int(document_id),
            "text": text,
            "label": label,
            "start_char": int(start_char),
            "end_char": int(end_char),
            "confidence": float(confidence),
            "model": model
        }

        response = _make_api_request('post', 'entities', json=entity_data)

        if response.status_code != 201:
            error_detail = response.text
            app.logger.error(f"Failed to create entity: {response.status_code} - {error_detail}")
            return jsonify({"status": "error", "message": f"Failed to create entity: {error_detail}"})

        result = response.json()
        entity_id = result.get("id")

        app.logger.info(f"Entity created successfully | ID: {entity_id}")

        return jsonify({
            "status": "success",
            "entity_id": entity_id,
            "entity": result
        })

    except Exception as e:
        app.logger.exception("Error creating entity")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/delete-entity', methods=['POST'])
def api_delete_entity():
    """Delete an entity."""
    try:
        data = request.json
        entity_id = data.get("entity_id")

        if not entity_id:
            return jsonify({"status": "error", "message": "Entity ID is required"})

        app.logger.info(f"Deleting entity: {entity_id}")

        response = _make_api_request('delete', f'entities/{entity_id}')

        if response.status_code != 200:
            app.logger.error(f"Failed to delete entity {entity_id}: {response.text}")
            return jsonify({"status": "error", "message": "Failed to delete entity"})

        app.logger.info(f"Entity {entity_id} deleted successfully")
        return jsonify({"status": "success"})

    except Exception as e:
        app.logger.exception("Error deleting entity")
        return jsonify({"status": "error", "message": str(e)})


# ============================================================================
# API ENDPOINTS - ANNOTATIONS
# ============================================================================

@app.route('/api/submit-annotation', methods=['POST'])
def api_submit_annotation():
    """Submit annotation feedback."""
    try:
        data = request.json
        entity_id = data.get("entity_id")
        is_correct = data.get("is_correct", False)
        task_id = data.get("task_id")

        app.logger.info(f"Submitting annotation feedback | Entity: {entity_id}, Correct: {is_correct}")

        annotation_data = {
            "entity_id": entity_id,
            "is_correct": is_correct,
            "task_id": task_id,
            "user_id": "ui_user",
            "notes": f"Feedback from task {task_id}" if task_id else None
        }

        # If user provided corrections
        if not is_correct and data.get("corrected_entity"):
            annotation_data["corrected_entity"] = {
                "text": data["corrected_entity"]["text"],
                "label": data["corrected_entity"]["label"],
                "start_char": data["corrected_entity"]["start_char"],
                "end_char": data["corrected_entity"]["end_char"]
            }

        app.logger.info(f"Sending annotation to API: {json.dumps(annotation_data)}")

        response = _make_api_request('post', 'annotations/submit', json=annotation_data)

        if response.status_code != 201:
            error_detail = response.text
            app.logger.error(f"Failed to submit annotation: {response.status_code} - {error_detail}")
            return jsonify({"status": "error", "message": f"Failed to submit annotation: {error_detail}"})

        app.logger.info(f"Annotation submitted successfully")
        return jsonify({"status": "success", "result": response.json()})

    except Exception as e:
        app.logger.exception("Error submitting annotation")
        return jsonify({"status": "error", "message": str(e)})


# ============================================================================
# API ENDPOINTS - MODELS
# ============================================================================

@app.route('/api/reapply-model', methods=['POST'])
def api_reapply_model():
    """Re-apply the current model to a document."""
    try:
        data = request.json
        document_id = data.get("document_id")

        if not document_id:
            return jsonify({"status": "error", "message": "Document ID is required"})

        app.logger.info(f"Re-applying model to document: {document_id}")

        # Delete all existing entities for this document
        entities_response = _make_api_request('get', f'entities?document_id={document_id}')

        if entities_response.status_code == 200:
            entities_data = entities_response.json()
            entities = entities_data.get("entities", [])

            for entity in entities:
                _make_api_request('delete', f"entities/{entity['id']}")

            app.logger.info(f"Deleted {len(entities)} existing entities")

        # Re-apply the model
        process_response = _make_api_request('post', 'process', json={"document_id": document_id})

        if process_response.status_code != 200:
            app.logger.error(f"Failed to re-apply model: {process_response.text}")
            return jsonify({"status": "error", "message": f"Failed to re-apply model: {process_response.text}"})

        result = process_response.json()
        app.logger.info(f"Model re-applied successfully | Entities found: {result.get('entities_found', 0)}")

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        app.logger.exception("Error re-applying model")
        return jsonify({"status": "error", "message": str(e)})


# ============================================================================
# API ENDPOINTS - LABELS
# ============================================================================

@app.route('/api/add-label', methods=['POST'])
def api_add_label():
    """Add a new label to the system."""
    try:
        data = request.json
        label_name = data.get("name")

        if not label_name:
            return jsonify({"status": "error", "message": "Label name is required"})

        app.logger.info(f"Adding new label: {label_name}")

        response = _make_api_request('post', 'labels', json={"name": label_name})

        if response.status_code != 201:
            error_detail = response.text
            app.logger.error(f"Failed to add label: {response.status_code} - {error_detail}")
            return jsonify({"status": "error", "message": f"Failed to add label: {error_detail}"})

        result = response.json()
        app.logger.info(f"Label added successfully: {result}")

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        app.logger.exception("Error adding label")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/update-label', methods=['PUT'])
def api_update_label():
    """Update an existing label."""
    try:
        data = request.json
        old_label = data.get("old_label")
        new_label = data.get("new_label")
        category = data.get("category", "Altro")

        if not old_label or not new_label:
            return jsonify({"status": "error", "message": "Old and new label names are required"})

        app.logger.info(f"Updating label: {old_label} -> {new_label}")

        response = _make_api_request('put', 'labels', json={
            "old_label": old_label,
            "new_label": new_label,
            "category": category
        })

        if response.status_code != 200:
            error_detail = response.text
            app.logger.error(f"Failed to update label: {response.status_code} - {error_detail}")
            return jsonify({"status": "error", "message": f"Failed to update label: {error_detail}"})

        result = response.json()
        app.logger.info(f"Label updated successfully: {result}")

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        app.logger.exception("Error updating label")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/delete-label', methods=['DELETE'])
def api_delete_label():
    """Delete a label from the system."""
    try:
        data = request.json
        label = data.get("label")

        if not label:
            return jsonify({"status": "error", "message": "Label name is required"})

        app.logger.info(f"Deleting label: {label}")

        response = _make_api_request('delete', 'labels', json={"label": label})

        if response.status_code != 200:
            error_detail = response.text
            app.logger.error(f"Failed to delete label: {response.status_code} - {error_detail}")
            return jsonify({"status": "error", "message": f"Failed to delete label: {error_detail}"})

        result = response.json()
        app.logger.info(f"Label deleted successfully: {result}")

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        app.logger.exception("Error deleting label")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/get-labels', methods=['GET'])
def api_get_labels():
    """Get all labels."""
    try:
        response = _make_api_request('get', 'labels')

        if response.status_code != 200:
            error_detail = response.text
            app.logger.error(f"Failed to get labels: {response.status_code} - {error_detail}")
            return jsonify({"status": "error", "message": f"Failed to get labels: {error_detail}"})

        labels = response.json()
        return jsonify({"status": "success", "labels": labels})

    except Exception as e:
        app.logger.exception("Error getting labels")
        return jsonify({"status": "error", "message": str(e)})


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, port=5001)
