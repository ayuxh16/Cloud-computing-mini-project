FROM python:3.11-slim

WORKDIR /app

COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY worker/worker.py /app/worker.py

CMD ["python", "/app/worker.py"]
