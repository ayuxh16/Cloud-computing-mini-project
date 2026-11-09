import os
import json
import time
import io
import requests
import boto3

from pathlib import Path
from botocore.config import Config
from botocore.exceptions import ClientError

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

CHECKPOINT_KEY = os.getenv(
    "CHECKPOINT_KEY",
    f"teacher/checkpoints/shard-{SHARD_INDEX}.jsonl"
)

CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "10"))

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

print(f"Starting checkpointed worker: shard {SHARD_INDEX}/{SHARD_COUNT}", flush=True)

# Download original command pool.
source = io.BytesIO()
s3.download_fileobj(INPUT_BUCKET, INPUT_KEY, source)
source.seek(0)

commands = [
    json.loads(line)
    for line in source.getvalue().decode("utf-8").splitlines()
    if line.strip()
]

# Preserve original modulo-based sharding.
shard = [
    (line_number, record)
    for line_number, record in enumerate(commands)
    if line_number % SHARD_COUNT == SHARD_INDEX
]

if MAX_RECORDS > 0:
    shard = shard[:MAX_RECORDS]

print(f"Commands assigned: {len(shard)}", flush=True)

shard_ids = {record["id"] for _, record in shard}

# Load previously saved checkpoint.
results_by_id = {}

try:
    saved = s3.get_object(
        Bucket=OUTPUT_BUCKET,
        Key=CHECKPOINT_KEY
    )

    checkpoint_text = saved["Body"].read().decode("utf-8")

    for line in checkpoint_text.splitlines():
        if not line.strip():
            continue

        item = json.loads(line)

        if item["id"] in shard_ids:
            results_by_id[item["id"]] = item

    print(
        f"Resuming from checkpoint: {len(results_by_id)} records",
        flush=True
    )

except ClientError as error:
    code = error.response.get("Error", {}).get("Code", "")

    if code not in ("NoSuchKey", "404", "NotFound"):
        raise

    print("No previous checkpoint found. Starting fresh.", flush=True)


def ordered_results():
    return [
        results_by_id[record["id"]]
        for _, record in shard
        if record["id"] in results_by_id
    ]


def save_checkpoint():
    output = "\n".join(
        json.dumps(item, ensure_ascii=False)
        for item in ordered_results()
    )

    if output:
        output += "\n"

    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=CHECKPOINT_KEY,
        Body=output.encode("utf-8"),
        ContentType="application/x-ndjson"
    )

    print(
        f"CHECKPOINT SAVED: {len(results_by_id)} records "
        f"at s3://{OUTPUT_BUCKET}/{CHECKPOINT_KEY}",
        flush=True
    )


session = requests.Session()
failed_records = []

for position, (line_number, record) in enumerate(shard, start=1):

    command_id = record["id"]

    # Never regenerate an already checkpointed command.
    if command_id in results_by_id:
        continue

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

    teacher_raw = None

    # Retry transient failures with exponential backoff.
    for attempt in range(1, 7):
        try:
            response = session.post(
                OLLAMA_URL,
                json=payload,
                timeout=(15, 240)
            )

            response.raise_for_status()

            teacher_raw = response.json()["response"]

            if not isinstance(teacher_raw, str):
                raise ValueError("Teacher response is not a string")

            break

        except (
            requests.exceptions.RequestException,
            ValueError,
            KeyError
        ) as error:

            print(
                f"Command {command_id}, attempt {attempt}/6 failed: "
                f"{str(error)[:300]}",
                flush=True
            )

            if attempt < 6:
                delay = min(60, 5 * (2 ** (attempt - 1)))
                print(f"Retrying in {delay} seconds", flush=True)
                time.sleep(delay)

    if teacher_raw is None:
        failed_records.append({
            "id": command_id,
            "text": command,
            "error": "All six inference attempts failed"
        })

        # Preserve successful responses even when a command fails.
        save_checkpoint()
        continue

    results_by_id[command_id] = {
        "id": command_id,
        "text": command,
        "teacher_raw": teacher_raw
    }

    completed = len(results_by_id)

    if completed % CHECKPOINT_INTERVAL == 0:
        save_checkpoint()

    if completed % 10 == 0 or completed == len(shard):
        print(
            f"Shard {SHARD_INDEX}: "
            f"{completed}/{len(shard)} completed",
            flush=True
        )

# Always save the last successful responses.
save_checkpoint()

# Save failed commands separately.
if failed_records:
    failure_text = "\n".join(
        json.dumps(item, ensure_ascii=False)
        for item in failed_records
    ) + "\n"

    s3.put_object(
        Bucket=OUTPUT_BUCKET,
        Key=f"teacher/failures/shard-{SHARD_INDEX}.jsonl",
        Body=failure_text.encode("utf-8"),
        ContentType="application/x-ndjson"
    )

    raise RuntimeError(
        f"{len(failed_records)} commands failed. "
        "Successful results are checkpointed. "
        "Rerun this worker to retry missing commands."
    )

# Only publish the final shard if every command succeeded.
if len(results_by_id) != len(shard):
    raise RuntimeError("Shard is incomplete; final output not published")

final_output = "\n".join(
    json.dumps(item, ensure_ascii=False)
    for item in ordered_results()
) + "\n"

s3.put_object(
    Bucket=OUTPUT_BUCKET,
    Key=OUTPUT_KEY,
    Body=final_output.encode("utf-8"),
    ContentType="application/x-ndjson"
)

print(f"SUCCESS: uploaded s3://{OUTPUT_BUCKET}/{OUTPUT_KEY}", flush=True)
print(f"Records uploaded: {len(results_by_id)}", flush=True)
