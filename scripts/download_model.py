from huggingface_hub import snapshot_download
import os

# Hugging Face model repo ID
MODEL_ID = "Sangeetha23/codebert-vuln-logic"

# Local directory where the model will be stored
TARGET_DIR = "models/vulnerability_logic_production"

# Create target directory if it does not exist
os.makedirs(TARGET_DIR, exist_ok=True)

# Download entire model repo from Hugging Face
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=TARGET_DIR,
    local_dir_use_symlinks=False
)

print("✅ Vulnerability model downloaded successfully")
