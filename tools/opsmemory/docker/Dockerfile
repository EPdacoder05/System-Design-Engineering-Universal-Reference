FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/inbox/processed

# Security: create a non-root system user with no shell and no home directory.
# All container processes run as this user — never as root.
RUN groupadd -r appuser && \
    useradd -r -g appuser -s /sbin/nologin appuser

# Transfer ownership of the application directory so appuser can read/write
# the inbox volume mount and any runtime artefacts.
RUN chown -R appuser:appuser /app

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Drop privileges: run as non-root from this point forward.
USER appuser

CMD ["sh", "-c", "python tools/opsmemory/scripts/migrate.py && uvicorn tools.opsmemory.api.app:app --host 0.0.0.0 --port 8000"]
