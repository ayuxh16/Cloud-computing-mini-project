import os
import json
import time
import io
import requests
import boto3

from pathlib import Path
from botocore.config import Config

SHARD_INDEX = int(os.environ["SHARD_INDEX"])
SHARD_COUNT = int(os.environ["SHARD_COUNT"])
MAX_RECORDS = int(os.getenv("MAX_RECORDS", "0"))

MODEL = os.getenv("MODEL", "phi3.5")
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

S3_ENDPOINT = os.getenv("S3_ENDPOINT", "http://localhost:8333")
INPUT_BUCKET = "datasets"
INPUT_KEY = "commands_pool.jsonl"
OUTPUT_BUCKET = "datasets"

OUTPUT_KEY = os.getenv(
    "OUTPUT_KEY",
    f"teacher/shard-{SHARD_INDEX}.jsonl"
)

if SHARD_COUNT < 1 or not 0 <= SHARD_INDEX < SHARD_COUNT:
    raise ValueError("Invalid SHARD_INDEX or SHARD_COUNT")

access_key = os.environ["AWS_ACCESS_KEY_ID"]
secret_key = os.environ["AWS_SECRET_ACCESS_KEY"]

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name="us-east-1",
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"}
    )
)

prompt_path = Path("prompts/teacher_prompt.txt")
if not prompt_path.exists():
    raise FileNotFoundError("Teacher prompt not found")

prompt_template = prompt_path.read_text(encoding="utf-8")

print(f"Starting worker: shard {SHARD_INDEX}/{SHARD_COUNT}")

# Download original command pool from object storage.
source = io.BytesIO()
s3.download_fileobj(INPUT_BUCKET, INPUT_KEY, source)
source.seek(0)

commands = [
    json.loads(line)
    for line in source.getvalue().decode("utf-8").splitlines()
    if line.strip()
]

# Shard by zero-based line index, not command ID.
shard = [
    (line_number, record)
    for line_number, record in enumerate(commands)
    if line_number % SHARD_COUNT == SHARD_INDEX
]

if MAX_RECORDS > 0:
    shard = shard[:MAX_RECORDS]

print(f"Commands assigned to this worker: {len(shard)}")

results = []

for position, (line_number, record) in enumerate(shard, start=1):
    command = record["text"]

    payload = {
        "model": MODEL,
        "prompt": prompt_template + "\n" + command,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 128
        }
    }

    # Retry transient errors. Never replace a teacher response manually.
    for attempt in range(1, 4):
        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=600
            )
            response.raise_for_status()

            teacher_raw = response.json()["response"]
            if not isinstance(teacher_raw, str):
                raise ValueError("Ollama returned a non-string response")

            break

        except Exception as error:
            print(
                f"Command {record['id']}, attempt {attempt}/3 failed: {error}",
                flush=True
            )
            if attempt == 3:
                raise
            time.sleep(3 * attempt)

    results.append({
        "id": record["id"],
        "text": command,
        "teacher_raw": teacher_raw
    })

    if position % 10 == 0 or position == len(shard):
        print(
            f"Shard {SHARD_INDEX}: {position}/{len(shard)} completed",
            flush=True
        )

# Upload the completed shard.
output = io.BytesIO()

for result in results:
    output.write(
        (json.dumps(result, ensure_ascii=False) + "\n").encode("utf-8")
    )

output.seek(0)

s3.upload_fileobj(
    output,
    OUTPUT_BUCKET,
    OUTPUT_KEY
)

print(f"SUCCESS: uploaded s3://{OUTPUT_BUCKET}/{OUTPUT_KEY}")
print(f"Records uploaded: {len(results)}")
