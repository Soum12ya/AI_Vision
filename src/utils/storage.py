import os
import json
import logging
from typing import Tuple, Dict, Any

# Assuming OUTPUT_DIR is defined in your config
from ..config import OUTPUT_DIR

def ensure_job_dirs(job_id: str) -> Tuple[str, str, str]:
    """
    Creates the necessary directory structure for a given job_id and returns the paths.

    This function ensures that each job has its own isolated set of folders for its
    output files, preventing any conflicts.

    Args:
        job_id: The unique identifier for the processing job.

    Returns:
        A tuple containing the paths to the processed images, intermediate results,
        and final results directories.
    """
    # Base directory for this specific job's output
    job_output_dir = os.path.join(OUTPUT_DIR, job_id)

    # Define the specific subdirectories we need
    processed_images_dir = os.path.join(job_output_dir, "processed_images")
    intermediate_results_dir = os.path.join(job_output_dir, "intermediate_results")
    final_results_dir = os.path.join(job_output_dir, "final_results")

    # Create all directories if they don't already exist
    os.makedirs(processed_images_dir, exist_ok=True)
    os.makedirs(intermediate_results_dir, exist_ok=True)
    os.makedirs(final_results_dir, exist_ok=True)

    return processed_images_dir, intermediate_results_dir, final_results_dir

def write_json(file_path: str, data: Dict[str, Any]):
    """
    Safely writes a dictionary to a JSON file.

    Args:
        file_path: The full path to the output JSON file.
        data: The dictionary data to write.
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to write JSON to {file_path}: {e}")

def read_json(file_path: str) -> Dict[str, Any]:
    """
    Safely reads data from a JSON file.

    Args:
        file_path: The full path to the JSON file to read.

    Returns:
        A dictionary with the file's content, or an empty dictionary if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"JSON file not found at {file_path}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {file_path}")
        return {}
    except Exception as e:
        logging.error(f"Failed to read JSON from {file_path}: {e}")
        return {}

