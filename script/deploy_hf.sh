#!/usr/bin/env bash
# Deploy current working directory to Hugging Face Spaces
set -euo pipefail

REPO_ID="Feng-X/ring-sizer"
IGNORE='[".venv/*", ".git/*", "__pycache__/*", "*.pyc", "output/*", "web_demo/uploads/*", "web_demo/results/*", "doc/*", ".claude/*", "input/*"]'

cd "$(dirname "$0")/.."

source .venv/bin/activate

python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(
    folder_path='.',
    repo_id='${REPO_ID}',
    repo_type='space',
    ignore_patterns=${IGNORE},
)
print('Deployed to https://huggingface.co/spaces/${REPO_ID}')
"
