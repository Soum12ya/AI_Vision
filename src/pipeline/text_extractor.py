import os
import logging
from typing import List, Dict
import pdfplumber
import re
from PIL import Image
import pytesseract
import camelot
import pandas as pd

# --- Helper Functions for Image Processing (OCR) ---

def ocr_image_to_text(image_path: str) -> str:
    """Performs OCR on an entire image and returns the text."""
    try:
        im = Image.open(image_path)
        return pytesseract.image_to_string(im)
    except Exception as e:
        logging.error(f"OCR failed for image {image_path}: {e}")
        return ""

def extract_schedule_from_image(image_path: str) -> List[Dict]:
    """
    Extracts a lighting schedule from an image using OCR.
    NOTE: This is a simplified approach and may be less accurate than Camelot on PDFs.
    It assumes a simple table structure.
    """
    logging.info(f"Attempting to extract schedule from IMAGE: {os.path.basename(image_path)}")
    text = ocr_image_to_text(image_path)
    lines = text.splitlines()
    
    # Heuristic to find the header row
    header_keywords = ["type mark", "lamp", "wattage", "voltage", "description"]
    header_index = -1
    headers = []
    for i, line in enumerate(lines):
        if all(keyword in line.lower() for keyword in ["type mark", "description"]): # Find a likely header
            headers = [h.strip().lower().replace(" ", "_") for h in line.split('\t')] # Split by tabs or multiple spaces
            if 'type_mark' not in headers: # try splitting by spaces
                 headers = [h.strip().lower().replace(" ", "_") for h in re.split(r'\s{2,}', line)]
            header_index = i
            break
            
    if header_index == -1:
        logging.warning("Could not find a clear header row in the OCR'd schedule.")
        return []

    # Process rows after the header
    rows = []
    for line in lines[header_index + 1:]:
        if not line.strip():
            continue
        
        values = re.split(r'\s{2,}', line.strip()) # Split by 2+ spaces
        if len(values) < 2: continue

        row_data = dict(zip(headers, values))
        
        # We only care about the columns specified in the project
        rows.append({
            "type": "table_row",
            "symbol": row_data.get("type_mark", ""),
            "description": row_data.get("description", ""),
            "mount": row_data.get("mounting", ""),
            "voltage": row_data.get("voltage", ""),
            "lumens": row_data.get("initial_nom._lumen_output", ""),
            "source_sheet": f"Lighting Schedule ({os.path.basename(image_path)})"
        })
    return rows


# --- Main Functions (Updated to handle both PDFs and Images) ---

def extract_general_notes(file_path: str) -> List[Dict]:
    """Extracts general notes from either a PDF or an Image."""
    notes = []
    filename = os.path.basename(file_path)
    
    text = ""
    if file_path.lower().endswith('.pdf'):
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                # Add source sheet info to each line
                notes.extend([
                    {"type": "note", "text": line.strip(), "source_sheet": f"{filename}_page_{i}"}
                    for line in page_text.splitlines()
                ])
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        text = ocr_image_to_text(file_path)
        notes.extend([
            {"type": "note", "text": line.strip(), "source_sheet": filename}
            for line in text.splitlines()
        ])

    # Filter for lines that look like notes
    filtered_notes = []
    for note in notes:
        line = note['text']
        if re.match(r"^\s*(\d+\.|[-*•])\s+", line) or "GENERAL NOTE" in line.upper():
            filtered_notes.append(note)

    # Deduplicate notes
    unique_notes, seen = [], set()
    for n in filtered_notes:
        k = n["text"].lower()
        if k not in seen:
            unique_notes.append(n)
            seen.add(k)
            
    return unique_notes

def extract_lighting_schedule(file_path: str) -> List[Dict]:
    """Extracts the lighting schedule from either a PDF or an Image."""
    if file_path.lower().endswith('.pdf'):
        logging.info(f"Attempting to extract schedule from PDF: {os.path.basename(file_path)}")
        try:
            tables = camelot.read_pdf(file_path, pages="all", flavor="stream")
            if not tables:
                logging.warning(f"Camelot found no tables in {file_path}")
                return []
            
            # Assume the largest table is the one we want
            df = tables[0].df
            
            # Clean up the dataframe headers
            df.columns = df.iloc[0]
            df = df[1:]
            df.columns = [str(c).strip().lower().replace("\n", " ").replace(" ", "_") for c in df.columns]
            
            rows = []
            for _, r in df.iterrows():
                row_dict = r.to_dict()
                if "type_mark" in row_dict and row_dict['type_mark']:
                    rows.append({
                        "type": "table_row",
                        "symbol": row_dict.get("type_mark", ""),
                        "description": row_dict.get("description", ""),
                        "mount": row_dict.get("mounting", ""),
                        "voltage": row_dict.get("voltage", ""),
                        "lumens": row_dict.get("initial_nom._lumen_output", ""),
                        "source_sheet": f"Lighting Schedule ({os.path.basename(file_path)})"
                    })
            return rows
        except Exception as e:
            logging.error(f"Camelot failed to process {file_path}: {e}")
            return []
            
    elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        return extract_schedule_from_image(file_path)
        
    return []


# --- Orchestrator Function ---

def extract_static_content(file_path: str) -> Dict[str, List[Dict]]:
    """
    Main orchestrator to extract all static content (notes and schedules)
    from a given file (PDF or Image).
    """
    print(f"Extracting content from: {os.path.basename(file_path)}")
    
    # You can decide which function to call based on the filename or content
    # For this project, we assume some files are for notes and others for schedules.
    
    notes = extract_general_notes(file_path)
    schedule = extract_lighting_schedule(file_path)
    
    return {"notes": notes, "schedule": schedule}