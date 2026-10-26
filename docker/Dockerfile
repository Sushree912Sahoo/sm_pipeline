FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Kolkata \
    SM_PIPELINE_ROOT=/app

# tzdata: Indian time zone; libgomp1: OpenMP runtime needed by xgboost
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-dashboard.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# The project code and data are NOT copied into the image: docker-compose mounts your
# sm_pipeline folder at /app (read-only), so the dashboard always sees the newest files.
EXPOSE 5000

# ONE worker on purpose: each worker would load its own copy of the two 27-million-row lookup tables.
CMD ["gunicorn", "--chdir", "/app/dashboard", "--workers", "1", "--threads", "4", \
     "--timeout", "600", "--bind", "0.0.0.0:5000", "app:app"]
