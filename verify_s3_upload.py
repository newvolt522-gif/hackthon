"""
Read-only verification of the SageMaker XGBoost training data in S3.
Does NOT modify or delete anything.
"""
import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"
BUCKET = "kiro-workshop-kiro"
EXPECTED_KEY = "sagemaker/xgboost-iris/train/train.csv"


def main():
    # 1. AWS Account ID
    sts = boto3.client("sts", region_name=REGION)
    account_id = sts.get_caller_identity()["Account"]
    print(f"1. AWS Account ID: {account_id}")

    # 2. Region actually in use by the session/client
    session = boto3.session.Session()
    s3 = boto3.client("s3", region_name=REGION)
    print(f"2. Region: {s3.meta.region_name} (session default: {session.region_name})")

    # 3. Does the bucket exist / is it accessible?
    try:
        s3.head_bucket(Bucket=BUCKET)
        bucket_ok = True
        print(f"3. Bucket '{BUCKET}': EXISTS and is accessible")
    except ClientError as e:
        bucket_ok = False
        code = e.response["Error"]["Code"]
        print(f"3. Bucket '{BUCKET}': NOT accessible (error {code})")
        if code == "404":
            print("   -> The bucket itself does not exist.")
            print(f"\n6. S3 URI (expected): s3://{BUCKET}/{EXPECTED_KEY}")
            return

    # 4 & 5. Does the exact key exist, and how big is it?
    exists = False
    size = None
    try:
        head = s3.head_object(Bucket=BUCKET, Key=EXPECTED_KEY)
        exists = True
        size = head["ContentLength"]
        print(f"4. Object '{EXPECTED_KEY}': EXISTS")
        print(f"5. Size: {size} bytes")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        print(f"4. Object '{EXPECTED_KEY}': NOT FOUND (error {code})")
        print("5. Size: n/a (object not found)")

    # 6. Full S3 URI (the expected/target path)
    print(f"6. S3 URI: s3://{BUCKET}/{EXPECTED_KEY}")

    # 7. If missing, figure out which prefix level is absent.
    if not exists and bucket_ok:
        print("\n7. Missing-path diagnosis:")
        parts = EXPECTED_KEY.split("/")
        # Build cumulative prefixes: sagemaker/, sagemaker/xgboost-iris/, ...
        cumulative = ""
        first_missing = None
        for i, part in enumerate(parts):
            cumulative = part if not cumulative else f"{cumulative}/{part}"
            is_last = i == len(parts) - 1
            probe = cumulative if is_last else cumulative + "/"
            resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=probe, MaxKeys=1)
            present = resp.get("KeyCount", 0) > 0
            label = "file" if is_last else "prefix"
            status = "present" if present else "MISSING"
            print(f"   - {label} '{probe}': {status}")
            if not present and first_missing is None:
                first_missing = probe

        if first_missing is not None:
            print(f"\n   First missing level: '{first_missing}'")

        # Show what actually IS in the bucket for context.
        print("\n   Objects actually present in the bucket (up to 20):")
        listing = s3.list_objects_v2(Bucket=BUCKET, MaxKeys=20)
        if listing.get("KeyCount", 0) == 0:
            print("   (bucket is empty)")
        else:
            for obj in listing["Contents"]:
                print(f"   - {obj['Key']} ({obj['Size']} bytes)")


if __name__ == "__main__":
    main()
