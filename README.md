# AI Vision for Emergency Lighting Detection

This project is an end-to-end AI pipeline that automates the process of electrical takeoff for emergency lighting fixtures from construction blueprints. It uses a computer vision model to detect lights, an OCR engine to read associated symbols and schedules, and a Large Language Model (LLM) to group and summarize the findings into a structured report.

---

## 🚀 Features

-   **Automated Detection:** A YOLOv8 model detects shaded emergency light fixtures on electrical drawings.
-   **Text Association:** Targeted OCR extracts nearby text symbols (e.g., "A1E", "W") for each fixture.
-   **Static Content Extraction:** Automatically parses Lighting Schedule tables and General Notes from PDFs.
-   **LLM-Powered Reasoning:** Uses GPT models to intelligently count, group, and describe fixtures based on the extracted rules.
-   **Asynchronous Processing:** A robust background job system using Celery and Redis handles long-running AI tasks without blocking the API.
-   **REST API:** Simple endpoints to upload a PDF and retrieve the final, structured JSON result.

---

## 🛠️ Tech Stack

-   **Backend:** Python, FastAPI
-   **AI / ML:** PyTorch, Ultralytics (YOLOv8), OpenAI API (GPT-4o-mini)
-   **OCR & PDF Processing:** Pytesseract, Camelot, pdfplumber
-   **Background Jobs:** Celery, Redis
-   **Deployment:** Render (Web Service, Background Worker, Redis)

---

## ⚙️ Setup and Local Installation

These instructions are for a Windows environment using WSL for Redis.

### Prerequisites

-   Python 3.9+
-   Git
-   WSL with a Linux distribution (e.g., Ubuntu) installed.
-   A trained YOLOv8 model file named `best.pt`.
