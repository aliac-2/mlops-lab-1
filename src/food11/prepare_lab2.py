
from pathlib import Path
import data as food11_data

# Project root: mlops-lab-1
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Use separate Lab 2 directories
LAB2_DIR = PROJECT_ROOT / "lab2_data"

food11_data.RAW_DIR = LAB2_DIR / "food11_raw"
food11_data.PROCESSED_DIR = LAB2_DIR / "food11_processed"
food11_data.MINI_DIR = LAB2_DIR / "food11_processed_mini"

# Reuse Lab 1 preprocessing
food11_data.prepare_dataset()

print("Lab 2 data preparation completed!")