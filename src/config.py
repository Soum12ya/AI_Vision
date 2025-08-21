import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Project Root Path ---
# This makes all other paths robust and independent of where the script is run
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Application Settings ---
PORT = int(os.getenv("PORT", "8000"))

# --- Directory and File Paths ---
# Build absolute paths from the project's base directory
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data", "input_pdfs"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.join(BASE_DIR, "outputs"))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "vision", "best.pt"))

# --- External Services ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")