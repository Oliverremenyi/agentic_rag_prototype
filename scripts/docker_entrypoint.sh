#!/bin/sh
# Container startup: wait for the ollama service to finish pulling both
# required models (docker-compose's `depends_on` only waits for the
# container to start, not for its `ollama pull` to finish), then build the
# FAISS index and launch the UI.
set -e

echo "Waiting for Ollama models to be ready at ${OLLAMA_BASE_URL}..."
uv run python -c "
import os
import sys
import time
import urllib.request
import json

base_url = os.environ.get('OLLAMA_BASE_URL', 'http://ollama:11434')
required = {os.environ.get('LLM_MODEL', 'llama3.2:3b'), os.environ.get('EMBEDDING_MODEL', 'nomic-embed-text')}
deadline = time.time() + 300

def is_satisfied(name, available):
    # An untagged model name (e.g. 'nomic-embed-text') resolves to ':latest'
    # server-side but is listed in /api/tags AS 'nomic-embed-text:latest'.
    return name in available or (':' not in name and f'{name}:latest' in available)

while time.time() < deadline:
    try:
        with urllib.request.urlopen(f'{base_url}/api/tags', timeout=5) as resp:
            data = json.load(resp)
        available = {m['name'] for m in data.get('models', [])}
        missing = {name for name in required if not is_satisfied(name, available)}
        if not missing:
            print('All required models available:', required)
            sys.exit(0)
        print('Waiting for models', missing, '...')
    except Exception as exc:
        print('Ollama not reachable yet:', exc)
    time.sleep(3)

print('Timed out waiting for Ollama models:', required)
sys.exit(1)
"

echo "Building FAISS index..."
uv run python -c "from src.rag.retriever import build_index; build_index()"

echo "Starting Streamlit UI..."
exec uv run streamlit run src/ui/streamlit_app.py --server.address 0.0.0.0
