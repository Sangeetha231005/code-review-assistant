#!/usr/bin/env python3
"""
Downloads vulnerability model ONLY if not already present locally.
Used by GitHub Actions to ensure model is available before analysis.
"""

from huggingface_hub import snapshot_download
import os
import sys

# Hugging Face model repo ID
MODEL_ID = "Sangeetha23/codebert-vuln-logic"

# Local directory where the model will be stored
TARGET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "models", "vulnerability_logic_production")

# Create target directory if it does not exist
os.makedirs(TARGET_DIR, exist_ok=True)

# ✅ CRITICAL: Check if model already exists
# Look for config.json as indicator of existing model
config_path = os.path.join(TARGET_DIR, "config.json")
model_path = os.path.join(TARGET_DIR, "model.safetensors")
pytorch_path = os.path.join(TARGET_DIR, "pytorch_model.bin")

# Check if any model files exist
model_exists = (
    os.path.exists(config_path) and 
    (os.path.exists(model_path) or os.path.exists(pytorch_path))
)

if model_exists:
    print("✅ Vulnerability model already exists — skipping download")
    print(f"   Location: {TARGET_DIR}")
    sys.exit(0)

print("⬇️ Downloading vulnerability model from Hugging Face...")
print(f"   Model: {MODEL_ID}")
print(f"   Target: {TARGET_DIR}")

# Download model with HF token if available
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    print("   Using HF_TOKEN for authentication")
else:
    print("   ⚠️ No HF_TOKEN found, downloading without authentication")

try:
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
        token=hf_token,
        resume_download=True,
        max_workers=4
    )
    
    # Verify download was successful
    if os.path.exists(config_path):
        print("✅ Vulnerability model downloaded successfully")
        print(f"   Config: {config_path}")
        
        # Check which model files were downloaded
        if os.path.exists(model_path):
            print(f"   Model file: model.safetensors")
        elif os.path.exists(pytorch_path):
            print(f"   Model file: pytorch_model.bin")
        else:
            print(f"   ⚠️ Warning: No model weights file found")
    else:
        print("❌ Download failed - config.json not found")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    sys.exit(1)
