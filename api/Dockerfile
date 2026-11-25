FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY api/requirements.txt /app/api/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/api/requirements.txt

COPY api /app/api
COPY dashboard /app/dashboard
COPY monitoreo /app/monitoreo
COPY scripts /app/scripts
COPY src /app/src
COPY tests/fixtures /app/tests/fixtures

USER 10001:10001

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
