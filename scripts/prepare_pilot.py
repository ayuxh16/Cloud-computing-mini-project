import os
import json
import re
import boto3
from botocore.config import Config
from pathlib import Path

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:8333",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name="us-east-1",
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"}
    )
)

valid_intents = {
    "alarm_set", "alarm_query", "calendar_set", "calendar_query",
    "calendar_remove", "weather_query", "datetime_query",
    "recommendation_events"
}
expected_keys = {"intent", "date", "time", "place"}

records = []

for shard in (0, 1):
    key = f"teacher/checkpoints/shard-{shard}.jsonl"
    print(f"Downloading {key}")
    response = s3.get_object(Bucket="datasets", Key=key)
    lines = response["Body"].read().decode("utf-8").splitlines()
    shard_records = [json.loads(x) for x in lines if x.strip()]
    print(f"Shard {shard}: {len(shard_records)} checkpointed records")
    records.extend(shard_records)

records.sort(key=lambda x: x["id"])

# Deduplicate before validating teacher labels.
seen = set()
unique = []

for record in records:
    normalized = re.sub(
        r"\s+", " ", record["text"].strip()
    ).casefold()
    if normalized in seen:
        continue
    seen.add(normalized)
    unique.append(record)

targets = []
rejected = 0

for record in unique:
    raw = record.get("teacher_raw")

    try:
        obj = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        rejected += 1
        continue

    if not isinstance(obj, dict) or set(obj) != expected_keys:
        rejected += 1
        continue

    if not isinstance(obj["intent"], str) or obj["intent"] not in valid_intents:
        rejected += 1
        continue

    valid = True
    for key in ("date", "time", "place"):
        value = obj[key]
        if value is not None and (
            not isinstance(value, str)
            or not value.strip()
            or value not in record["text"]
        ):
            valid = False
            break

    if not valid:
        rejected += 1
        continue

    targets.append({
        "id": record["id"],
        "text": record["text"],
        "teacher_raw": raw
    })

Path("results").mkdir(exist_ok=True)

with open(
    "results/teacher_targets_pilot.jsonl",
    "w",
    encoding="utf-8",
    newline="\n"
) as f:
    for item in targets:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("\nPILOT DATASET PREPARATION")
print(f"Raw checkpoint records: {len(records)}")
print(f"Unique commands: {len(unique)}")
print(f"Rejected records: {rejected}")
print(f"Valid pilot targets: {len(targets)}")
print("Output: results/teacher_targets_pilot.jsonl")

if not targets:
    raise SystemExit("No valid training targets found.")
