import os
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

print("Checking S3 connection...")

response = s3.list_buckets()

buckets = [bucket["Name"] for bucket in response["Buckets"]]
print("Existing buckets:", buckets)

for name in ["datasets", "models"]:
    if name not in buckets:
        s3.create_bucket(Bucket=name)
        print("Created bucket:", name)
    else:
        print("Bucket already exists:", name)

file_path = Path("data/commands_pool.jsonl")

if not file_path.exists():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

s3.upload_file(
    str(file_path),
    "datasets",
    "commands_pool.jsonl"
)

print("Uploaded:", file_path)

obj = s3.head_object(
    Bucket="datasets",
    Key="commands_pool.jsonl"
)

print("Upload verified!")
print("Object size:", obj["ContentLength"], "bytes")
print("S3 connection successful.")
