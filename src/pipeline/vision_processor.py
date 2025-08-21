from typing import List, Dict
import os
import logging
import pdfplumber
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
from ultralytics import YOLO

# Assuming MODEL_PATH is defined in your config
from ..config import MODEL_PATH

# --- Main Functions ---

def pdf_to_images(pdf_path: str, out_dir: str) -> List[str]:
    """Converts each page of a PDF to a PNG image."""
    paths = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # Using a higher resolution for better OCR and detection quality
                img = page.to_image(resolution=300)
                out = os.path.join(out_dir, f"page_{i:03d}.png")
                img.save(out, format="PNG")
                paths.append(out)
    except Exception as e:
        logging.error(f"Failed to convert PDF {pdf_path} to images: {e}")
    return paths

def _heuristic_rectangles(img):
    """Fallback function to detect rectangles using traditional CV methods."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mor = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)
    cnts, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 200:
            continue
        x, y, w, h = cv2.boundingRect(c)
        rect_area = w*h
        fill_ratio = area/(rect_area+1e-6)
        aspect = w/(h+1e-6)
        if 0.6 < fill_ratio < 1.2 and 0.8 < aspect < 5 and 200 < rect_area < 50000:
            dets.append({"bbox":[float(x),float(y),float(x+w),float(y+h)], "conf":float(min(0.6+0.4*fill_ratio,0.95))})
    return dets

def detect_on_image(image_path: str, model_path: str = MODEL_PATH) -> List[Dict]:
    """Detects objects on a single image using YOLO or a heuristic fallback."""
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not read image at {image_path}")
        return []
        
    dets = []
    try:
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            model = YOLO(model_path)
            res = model.predict(img, verbose=False)
            for r in res:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(float, b.xyxy[0])
                    conf = float(b.conf[0])
                    dets.append({"bbox":[x1,y1,x2,y2],"conf":conf})
    except Exception as e:
        logging.error(f"YOLO model prediction failed: {e}")
        pass
    
    if not dets:
        logging.warning("YOLO found no detections, attempting heuristic fallback.")
        dets = _heuristic_rectangles(img)
    return dets

def associate_symbols_with_detections(image_path: str, detections: List[Dict]) -> List[Dict]:
    """Associates text symbols with detected bounding boxes using OCR."""
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    updated_detections = []
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        
        expansion_px = 60
        search_x1 = max(0, x1 - expansion_px)
        search_y1 = max(0, y1 - expansion_px)
        search_x2 = min(w, x2 + expansion_px)
        search_y2 = min(h, y2 + expansion_px)
        
        search_crop = img[search_y1:search_y2, search_x1:search_x2]
        
        try:
            gray_crop = cv2.cvtColor(search_crop, cv2.COLOR_BGR2GRAY)
            thresh_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            custom_config = r'-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ- --psm 6'
            ocr_text = pytesseract.image_to_string(thresh_crop, config=custom_config)
            
            potential_symbols = re.findall(r'\b[A-Z0-9-]{1,4}\b', ocr_text.strip())
            
            det['symbol'] = potential_symbols[0] if potential_symbols else None
            
        except Exception as e:
            logging.error(f"OCR processing failed for a detection: {e}")
            det['symbol'] = None
            
        updated_detections.append(det)
        
    return updated_detections

def run_vision_pipeline(image_path: str) -> List[Dict]:
    """The main orchestrator for the vision processing pipeline."""
    logging.info(f"Starting vision pipeline for {os.path.basename(image_path)}...")
    
    detections = detect_on_image(image_path)
    if not detections:
        logging.warning("No light fixtures were detected.")
        return []
    
    logging.info(f"Detected {len(detections)} potential fixtures.")
    
    detections_with_symbols = associate_symbols_with_detections(image_path, detections)
    
    logging.info("Finished vision pipeline.")
    return detections_with_symbols

def annotate_and_save(image_path: str, detections: List[Dict], out_path: str):
    """Draws bounding boxes and symbols on an image and saves it."""
    img = cv2.imread(image_path)
    for d in detections:
        x1,y1,x2,y2 = map(int, d["bbox"])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"{d.get('symbol','[No Symbol]')} {d.get('conf',0):.2f}"
        cv2.putText(img, label, (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.imwrite(out_path, img)
