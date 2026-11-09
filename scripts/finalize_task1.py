import os
import json
import hashlib
import re
from pathlib import Path

import boto3
from botocore.config import Config

BUCKET = "datasets"
ENDPOINT = "http://localhost:8333"

SHARDS = [
    ("teacher/shard-0.jsonl", "results/shard-0.jsonl"),
    ("teacher/shard-1.jsonl", "results/shard-1.jsonl"),
]

VALID_INTENTS = {
    "alarm_set", "alarm_query", "calendar_set", "calendar_query",
    "calendar_remove", "weather_query", "datetime_query",
    "recommendation_events",
}

EXPECTED_KEYS = {"intent", "date", "time", "place"}
SLOT_KEYS = {"date", "time", "place"}

Path("results").mkdir(exist_ok=True)

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name="us-east-1",
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
    ),
)

# Download and verify both complete shards.
all_records = []

for s3_key, local_path in SHARDS:
    print(f"Downloading s3://{BUCKET}/{s3_key}")
    s3.download_file(BUCKET, s3_key, local_path)

    with open(local_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Downloaded {len(records)} records")

    if len(records) != 1500:
        raise RuntimeError(
            f"{s3_key}: expected 1500 records, got {len(records)}"
        )

    all_records.extend(records)

if len(all_records) != 3000:
    raise RuntimeError("Expected exactly 3000 records.")

# Validate record structure and IDs.
ids = []

for record in all_records:
    if not isinstance(record, dict):
        raise RuntimeError("Non-object record found.")

    if not isinstance(record.get("id"), str):
        raise RuntimeError("Missing or invalid record ID.")

    if not isinstance(record.get("text"), str):
        raise RuntimeError(f"Invalid text in {record.get('id')}")

    if not isinstance(record.get("teacher_raw"), str):
        raise RuntimeError(f"Missing teacher_raw in {record['id']}")

    ids.append(record["id"])

if len(set(ids)) != 3000:
    raise RuntimeError("Duplicate or missing IDs across shards.")

all_records.sort(key=lambda r: r["id"])

# Preserve all original teacher responses.
with open(
    "results/teacher_merged.jsonl",
    "w",
    encoding="utf-8",
    newline="\n",
) as f:
    for record in all_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Deduplicate BEFORE label validation.
# Keep the lexicographically smallest ID.
counts = {
    "raw_records": len(all_records),
    "duplicate_commands": 0,
    "invalid_teacher_json": 0,
    "wrong_keys": 0,
    "invalid_intent": 0,
    "invalid_slot_type": 0,
    "slot_not_present_exactly": 0,
    "cleaned_records": 0,
}

unique_records = []
seen_commands = set()

for record in all_records:
    normalized = re.sub(
        r"\s+", " ", record["text"].strip()
    ).casefold()

    if normalized in seen_commands:
        counts["duplicate_commands"] += 1
        continue

    seen_commands.add(normalized)
    unique_records.append(record)

print(f"Unique commands: {len(unique_records)}")

# Validate teacher outputs.
cleaned = []
teacher_targets = []

for record in unique_records:
    teacher_raw = record["teacher_raw"]

    try:
        prediction = json.loads(teacher_raw)
    except (json.JSONDecodeError, TypeError):
        counts["invalid_teacher_json"] += 1
        continue

    if (
        not isinstance(prediction, dict)
        or set(prediction.keys()) != EXPECTED_KEYS
    ):
        counts["wrong_keys"] += 1
        continue

    intent = prediction["intent"]

    if not isinstance(intent, str) or intent not in VALID_INTENTS:
        counts["invalid_intent"] += 1
        continue

    text = record["text"]
    slot_type_invalid = False
    slot_missing = False

    for key in SLOT_KEYS:
        value = prediction[key]

        if value is not None and (
            not isinstance(value, str) or not value.strip()
        ):
            slot_type_invalid = True
            break

        if value is not None and value not in text:
            slot_missing = True
            break

    if slot_type_invalid:
        counts["invalid_slot_type"] += 1
        continue

    if slot_missing:
        counts["slot_not_present_exactly"] += 1
        continue

    cleaned.append({
        "id": record["id"],
        "text": text,
        "intent": intent,
        "date": prediction["date"],
        "time": prediction["time"],
        "place": prediction["place"],
    })

    # Preserve exact teacher response; do not reconstruct JSON.
    teacher_targets.append({
        "id": record["id"],
        "text": text,
        "teacher_raw": teacher_raw,
    })

cleaned.sort(key=lambda r: r["id"])
teacher_targets.sort(key=lambda r: r["id"])

counts["cleaned_records"] = len(cleaned)

# Write parsed labels.
cleaned_path = Path("results/cleaned_train.jsonl")

with cleaned_path.open(
    "w", encoding="utf-8", newline="\n"
) as f:
    for record in cleaned:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Write original teacher responses for sequence-level KD.
targets_path = Path("results/teacher_targets.jsonl")

with targets_path.open(
    "w", encoding="utf-8", newline="\n"
) as f:
    for record in teacher_targets:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# SHA256 checksums.
digest = hashlib.sha256(cleaned_path.read_bytes()).hexdigest()
targets_digest = hashlib.sha256(targets_path.read_bytes()).hexdigest()

with open("results/train.sha256", "w", encoding="utf-8") as f:
    f.write(f"{digest}  cleaned_train.jsonl\n")
    f.write(f"{targets_digest}  teacher_targets.jsonl\n")

# Cleaning report.
report = {
    **counts,
    "cleaned_sha256": digest,
    "teacher_targets_sha256": targets_digest,
    "validation_used_for_training": False,
    "duplicate_policy": (
        "Deduplicate before validation; retain lexicographically smallest ID."
    ),
    "slot_policy": "Exact case-sensitive substring of original command.",
    "distillation_target": "Original teacher_raw response without reconstruction.",
}

with open("results/cleaning_report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

# Upload all final outputs.
uploads = [
    ("results/teacher_merged.jsonl", "teacher/teacher_merged.jsonl"),
    ("results/cleaned_train.jsonl", "teacher/cleaned_train.jsonl"),
    ("results/teacher_targets.jsonl", "teacher/teacher_targets.jsonl"),
    ("results/cleaning_report.json", "teacher/cleaning_report.json"),
    ("results/train.sha256", "teacher/train.sha256"),
]

for local_path, s3_key in uploads:
    print(f"Uploading {s3_key}")
    s3.upload_file(local_path, BUCKET, s3_key)

print("\n========== TASK 1 CLEANING REPORT ==========")
for key, value in counts.items():
    print(f"{key}: {value}")

print(f"\nCleaned SHA256: {digest}")
print(f"Teacher targets SHA256: {targets_digest}")
print("\nTASK 1 FINALIZATION SUCCESSFUL.")
