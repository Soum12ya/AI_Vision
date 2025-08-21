import os
import shutil
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse

# Assuming your config and tasks are set up correctly in these locations
from ..config import DATA_DIR
from ..tasks.background_jobs import process_pdf_task
from ..utils.storage import ensure_job_dirs, read_json # Note: Renamed for clarity

# Initialize the router
router = APIRouter(prefix="/blueprints", tags=["blueprints"])

@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Handles PDF file uploads, saves the file to a unique directory,
    and dispatches a background processing task.
    """
    # 1. Input Validation: Ensure the uploaded file is a PDF.
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 2. Unique Job Creation: Generate a unique ID for this upload.
    # This is crucial to prevent filename collisions and keep jobs isolated.
    job_id = str(uuid.uuid4())
    original_filename = file.filename
    
    # Create a dedicated directory for this job's files.
    job_data_dir = os.path.join(DATA_DIR, job_id)
    os.makedirs(job_data_dir, exist_ok=True)
    
    # Define the full path for the saved PDF.
    pdf_path = os.path.join(job_data_dir, original_filename)

    # 3. Save the File: Copy the uploaded file to the server.
    try:
        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        logging.error(f"Failed to save uploaded file for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not save file.")
    finally:
        file.file.close()

    # 4. Dispatch Background Task: Enqueue the processing task with Celery.
    # We pass the job_id and the full path to the file.
    logging.info(f"Dispatching task for job_id: {job_id}")
    task = process_pdf_task.delay(job_id=job_id, pdf_path=pdf_path)

    # 5. Return Immediate Response: Let the user know the job has started.
    return JSONResponse({
        "status": "uploaded",
        "original_filename": original_filename,
        "job_id": job_id, # The client will use this ID to poll for the result
        "message": "Processing started in background.",
        "task_id": task.id # Celery's internal task ID for debugging
    })

@router.get("/result")
async def get_result(job_id: str = Query(..., description="The unique ID of the processing job.")):
    """
    Retrieves the status or the final result of a processing job
    using the job_id provided by the /upload endpoint.
    """
    # 1. Locate Job Directories: Find the directories for this specific job.
    # This assumes you have a helper function `ensure_job_dirs` in your utils.
    try:
        _, inter_dir, final_dir = ensure_job_dirs(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No job found with ID: {job_id}")

    # 2. Check Status File: Look for the status JSON to see the job's state.
    status_path = os.path.join(inter_dir, "status.json")
    if not os.path.exists(status_path):
        # If no status file, assume it's still queued or just started.
        return JSONResponse({
            "job_id": job_id,
            "status": "processing",
            "message": "Processing is in progress. Please try again later."
        })

    status_data = read_json(status_path)
    current_status = status_data.get("status", "processing")

    # 3. Handle In-Progress Status: If not complete, return a waiting message.
    if current_status != "complete":
        return JSONResponse({
            "job_id": job_id,
            "status": current_status,
            "message": "Processing is still in progress. Please try again later."
        })

    # 4. Retrieve and Return Final Result: If complete, find and return the result file.
    final_path = os.path.join(final_dir, "result.json")
    if not os.path.exists(final_path):
        # This case might happen if the job completed but failed to write the final file.
        logging.error(f"Job {job_id} is marked complete, but result.json is missing.")
        raise HTTPException(status_code=404, detail="Result not found, though processing is complete.")
    
    final_result = read_json(final_path)

    return JSONResponse({
        "job_id": job_id,
        "status": "complete",
        "result": final_result
    })
