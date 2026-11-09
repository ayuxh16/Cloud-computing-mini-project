import boto3
import os
from botocore.config import Config

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

s3.download_file(
    "datasets",
    "teacher/test-shard-0.jsonl",
    "results/test-shard-0.jsonl"
)

print("Downloaded test shard successfully.")
