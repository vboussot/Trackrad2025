from huggingface_hub import hf_hub_download
import os
import shutil

files = [
    "SAM2.1_b+_finetune.pt",
]

for file in files:
    cached_path = hf_hub_download(repo_id="vboussot/Trackrad2025", filename=file, repo_type="model")
    local_path = os.path.join(".", file)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    shutil.copy(cached_path, local_path)
