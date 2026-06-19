# Single image used for both the API and the dashboard (commands differ).
# Build from the REPO ROOT:  docker build -f deploy/Dockerfile -t netflix-recsys .
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements.txt

# Code + trained artifacts (dataset.npz, models, bundled movie_titles.csv).
# The ~2 GB raw Netflix text files are NOT needed at serving time.
COPY config.py /app/config.py
COPY src /app/src
COPY deploy /app/deploy
COPY data/dataset.npz /app/data/dataset.npz
COPY data/movie_titles.csv /app/data/movie_titles.csv
COPY outputs/models /app/outputs/models

EXPOSE 8000 8501
# Default: API. docker-compose overrides the command for the dashboard.
CMD ["uvicorn", "deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
