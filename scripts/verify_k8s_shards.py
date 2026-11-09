import json
import boto3
from botocore.config import Config

s3 = boto3.client(
    "s3",
    endpoint_url="http://localhost:8333",
    region_name="us-east-1",
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"}
    )
)

keys = [
    "teacher/test-shard-0-k8s.jsonl",
    "teacher/test-shard-1-k8s.jsonl"
]

expected_keys = {"intent", "date", "time", "place"}

for key in keys:
    response = s3.get_object(Bucket="datasets", Key=key)
    content = response["Body"].read().decode("utf-8")
    records = [json.loads(line) for line in content.splitlines() if line.strip()]

    print(f"\nObject: s3://datasets/{key}")
    print(f"Records: {len(records)}")

    for record in records:
        prediction = json.loads(record["teacher_raw"])
        valid = set(prediction.keys()) == expected_keys

        print(f"ID: {record['id']}")
        print(f"Command: {record['text']}")
        print(f"Teacher JSON: {json.dumps(prediction)}")
        print(f"Exactly four keys: {valid}")

        if not valid:
            raise ValueError(f"Invalid teacher output for {record['id']}")

    if len(records) != 2:
        raise ValueError(f"Expected 2 records in {key}")

print("\nSUCCESS: Both Kubernetes shard outputs verified.")
