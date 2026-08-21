FROM python:3.11-slim

WORKDIR /app

# ── Python dependencies ──
# Install PyTorch CPU-only FIRST to avoid downloading ~2GB of CUDA libraries.
# sentence-transformers will detect torch is already installed and skip it.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Pre-download the embedding model at build time to avoid cold-start ──
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ── Copy application code ──
COPY . .

# ── Default entrypoint: run the FastAPI server ──
# Override in docker-compose for the worker service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
