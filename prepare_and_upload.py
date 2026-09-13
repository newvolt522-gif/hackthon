"""
Prepare Iris training data in SageMaker XGBoost CSV format, create an S3
bucket in us-west-2, and upload the training file.
"""
import csv
import time

import boto3
from botocore.exceptions import ClientError
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

REGION = "us-west-2"
BUCKET = "kiro-workshop-kiro"
TRAIN_FILE = "train.csv"
S3_KEY = "iris/train.csv"


def build_training_csv():
    """Create ~120-row training CSV: no header, first column = numeric label."""
    data = load_iris()
    X, y = data.data, data.target
    # 150 rows total; 0.8 train split -> 120 training rows.
    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=120, random_state=42, stratify=y
    )

    with open(TRAIN_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        for label, features in zip(y_train, X_train):
            # SageMaker XGBoost: label first, then features, no header.
            writer.writerow([int(label)] + [f"{v:.4f}" for v in features])

    print(f"Wrote {len(y_train)} rows to {TRAIN_FILE} (no header, label first).")
    return len(y_train)


def ensure_bucket(s3):
    """Create the bucket in us-west-2 (requires LocationConstraint)."""
    try:
        s3.create_bucket(
            Bucket=BUCKET,
            CreateBucketConfiguration={"LocationConstraint": REGION},
        )
        print(f"Created bucket: {BUCKET}")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            print(f"Bucket already exists ({code}), reusing: {BUCKET}")
        else:
            raise
    # Wait until the bucket is available before uploading.
    s3.get_waiter("bucket_exists").wait(Bucket=BUCKET)


def main():
    build_training_csv()

    s3 = boto3.client("s3", region_name=REGION)
    ensure_bucket(s3)

    s3.upload_file(TRAIN_FILE, BUCKET, S3_KEY)
    s3_uri = f"s3://{BUCKET}/{S3_KEY}"
    print(f"Uploaded training data to: {s3_uri}")
    print(f"\nS3 path: {s3_uri}")


if __name__ == "__main__":
    main()
