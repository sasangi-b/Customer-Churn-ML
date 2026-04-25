# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Working directory inside container ────────────────────────────────────────
WORKDIR /app

# ── Install dependencies first (cached layer) ─────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────────────────────
COPY . .

# ── Copy best model into the fixed path inference.py expects ──────────────────
# Run scripts/promote_model.py before building to populate models/best_model/
COPY models/best_model /app/model

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start the server ──────────────────────────────────────────────────────────
CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]