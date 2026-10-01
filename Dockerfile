FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY src ./src
COPY data/docs ./data/docs
COPY data/toc ./data/toc
COPY data/docs_manifest.json ./data/docs_manifest.json
COPY eval ./eval
COPY loadtest ./loadtest
COPY scripts/docker_entrypoint.sh ./scripts/docker_entrypoint.sh
RUN chmod +x ./scripts/docker_entrypoint.sh

ENV OLLAMA_BASE_URL=http://ollama:11434 \
    PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["./scripts/docker_entrypoint.sh"]
