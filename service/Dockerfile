FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu torch

COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service/app.py .

# Include the trained student model inside the image
COPY models/student-image/ /models/student/

ENV MODEL_DIR=/models/student
ENV MODEL_VERSION=pythia-70m-student-pilot
ENV MAX_NEW_TOKENS=96

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
