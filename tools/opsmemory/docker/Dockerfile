FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/inbox/processed

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["sh", "-c", "python tools/opsmemory/scripts/migrate.py && uvicorn tools.opsmemory.api.app:app --host 0.0.0.0 --port 8000"]
