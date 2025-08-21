import os
import logging
from celery import Celery

# --- Configuration and Utilities ---
from ..config import REDIS_URL, MODEL_PATH
from ..utils.storage import ensure_job_dirs, write_json

# --- Core AI Pipeline Modules ---
from ..pipeline.vision_processor import pdf_to_images, run_vision_pipeline, annotate_and_save
from ..pipeline.text_extractor import extract_static_content
from ..pipeline.llm_grouper import group_and_summarize_with_llm

# Initialize Celery
celery_app = Celery("ai_vision_takeoff", broker=REDIS_URL, backend=REDIS_URL)

# Configure logging for background tasks
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@celery_app.task
def process_pdf_task(job_id: str, pdf_path: str):
    """
    The main background task that orchestrates the entire AI pipeline for a given PDF.
    """
    logging.info(f"Starting processing for job_id: {job_id}")

    # 1. Setup Directories & Initial Status
    try:
        pimg_dir, inter_dir, final_dir = ensure_job_dirs(job_id)
        status_path = os.path.join(inter_dir, "status.json")
        write_json(status_path, {"status": "processing", "job_id": job_id, "message": "Converting PDF to images..."})
    except Exception as e:
        logging.error(f"Failed to set up directories for job {job_id}: {e}")
        return {"job_id": job_id, "status": "failed", "error": "Directory setup failed."}

    # 2. PDF to Image Conversion
    logging.info(f"[{job_id}] Step 1: Rasterizing PDF to images.")
    image_paths = pdf_to_images(pdf_path, str(pimg_dir))
    if not image_paths:
        write_json(status_path, {"status": "failed", "job_id": job_id, "message": "PDF could not be converted to images."})
        return {"job_id": job_id, "status": "failed", "error": "PDF conversion failed."}

    # 3. Static Content Extraction (The "Rulebook")
    # **IMPROVEMENT**: We now specifically extract from the PDF, which is more reliable for tables.
    logging.info(f"[{job_id}] Step 2: Extracting static content (schedules, notes).")
    write_json(status_path, {"status": "processing", "job_id": job_id, "message": "Extracting schedules and notes..."})
    static_content = extract_static_content(pdf_path)
    schedule = static_content.get("schedule", [])
    notes = static_content.get("notes", [])
    
    if not schedule:
        logging.warning(f"[{job_id}] Could not find or parse the lighting schedule from the PDF.")
    
    write_json(os.path.join(inter_dir, "rulebook.json"), {"notes": notes, "schedule": schedule})

    # 4. Vision Processing (Detection + Symbol Association)
    logging.info(f"[{job_id}] Step 3: Running vision detection and OCR on {len(image_paths)} images.")
    
    # **IMPROVEMENT**: Check if the model file exists and is trained.
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1000: # Check if file is tiny
        logging.error(f"[{job_id}] CRITICAL: Model file at {MODEL_PATH} is missing or not trained. Aborting vision processing.")
        write_json(status_path, {"status": "failed", "job_id": job_id, "message": "YOLO model file is missing or untrained."})
        return {"job_id": job_id, "status": "failed", "error": "Model not found."}

    all_detections = []
    for idx, img_path in enumerate(image_paths):
        # **IMPROVEMENT**: Skip pages that are unlikely to have floor plans (like schedules/notes).
        if "schedule" in os.path.basename(img_path).lower() or "legend" in os.path.basename(img_path).lower():
            continue

        write_json(status_path, {"status": "processing", "job_id": job_id, "message": f"Analyzing image {idx + 1}/{len(image_paths)}..."})
        detections_with_symbols = run_vision_pipeline(img_path)
        all_detections.extend(detections_with_symbols)
        
        if detections_with_symbols:
            ann_out = os.path.join(pimg_dir, f"page_{idx:03d}_annotated.png")
            annotate_and_save(img_path, detections_with_symbols, ann_out)

    write_json(os.path.join(inter_dir, "all_detections.json"), {"detections": all_detections})

    # 5. LLM Grouping and Summarization
    if not all_detections:
        logging.warning(f"[{job_id}] No detections were found. Skipping LLM grouping.")
        final_summary = {}
    else:
        logging.info(f"[{job_id}] Step 4: Sending {len(all_detections)} detections to LLM for grouping.")
        write_json(status_path, {"status": "processing", "job_id": job_id, "message": "Grouping and summarizing results with AI..."})
        final_summary = group_and_summarize_with_llm(all_detections, schedule)

    # 6. Save Final Result and Update Status
    logging.info(f"[{job_id}] Step 5: Saving final result.")
    final_path = os.path.join(final_dir, "result.json")
    write_json(final_path, final_summary)

    write_json(status_path, {"status": "complete", "job_id": job_id})
    logging.info(f"Successfully completed processing for job_id: {job_id}")
    
    return {"job_id": job_id, "status": "complete"}
