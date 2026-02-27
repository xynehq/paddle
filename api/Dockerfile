FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

COPY api /app/api

EXPOSE 8088

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8088"]
