"""
Configuration management for Azure OCR testing project
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"
OCR_RESULTS_DIR = OUTPUT_DIR / "ocr_results"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, CACHE_DIR, OCR_RESULTS_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Azure Computer Vision Configuration
AZURE_CV_ENDPOINT = os.getenv("AZURE_CV_ENDPOINT")
AZURE_CV_API_KEY = os.getenv("AZURE_CV_API_KEY")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
# OpenRouter LLM Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3.5-sonnet")

# Processing Settings
PDF_DPI = int(os.getenv("PDF_DPI", "450"))
IMAGE_FORMAT = os.getenv("IMAGE_FORMAT", "PNG")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(DATA_DIR / "ocr_testing.log"))

# Validation
if not AZURE_CV_ENDPOINT:
    raise ValueError("❌ AZURE_CV_ENDPOINT not set in .env file")

if not AZURE_CV_API_KEY:
    raise ValueError("❌ AZURE_CV_API_KEY not set in .env file")

print(f"✅ Configuration loaded successfully")
print(f"   Azure Endpoint: {AZURE_CV_ENDPOINT}")
print(f"   Database: {'Configured' if DATABASE_URL else 'Not configured'}")
print(f"   Input Directory: {INPUT_DIR}")
print(f"   Output Directory: {OUTPUT_DIR}")