"""
Copy existing training data to a target bucket using boto3 (no AWS CLI).

Safety:
- Verifies the target bucket exists AND is accessible before doing anything.
- If the bucket is missing or access is denied, STOP: do not create a bucket,
  do not modify any other S3 data.
- Never deletes the source object.
"""
import sys

import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"

SRC_BUCKET = "kiro-workshop-kiro"
SRC_KEY = "iris/train.csv"

DST_BUCKET = "ntpc-youbike-hackathon-2026"
DST_KEY = "sagemaker/xgboost-iris/train/train.csv"


def main():
    s3 = boto3.client("s3", region_name=REGION)

    # 1. Confirm target bucket exists and this account can access it.
    try:
        s3.head_bucket(Bucket=DST_BUCKET)
        print(f"Target bucket '{DST_BUCKET}': EXISTS and accessible.")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        http = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        print(f"Target bucket '{DST_BUCKET}': NOT usable (code={code}, http={http}).")
        if code in ("404", "NoSuchBucket"):
            print("Reason: bucket does not exist.")
        elif code in ("403", "AccessDenied"):
            print("Reason: access denied / no permission (bucket may belong to another account).")
        else:
            print("Reason: unexpected error.")
        print("STOPPING. No bucket created, no S3 data modified.")
        sys.exit(1)

    # Confirm the source object still exists (do not delete it).
    try:
        src_head = s3.head_object(Bucket=SRC_BUCKET, Key=SRC_KEY)
        print(f"Source object s3://{SRC_BUCKET}/{SRC_KEY}: EXISTS "
              f"({src_head['ContentLength']} bytes).")
    except ClientError as e:
        print(f"Source object s3://{SRC_BUCKET}/{SRC_KEY} not found: "
              f"{e.response['Error']['Code']}")
        print("STOPPING. Nothing modified.")
        sys.exit(1)

    # 3. Copy with copy_object (server-side copy; source is left untouched).
    s3.copy_object(
        Bucket=DST_BUCKET,
        Key=DST_KEY,
        CopySource={"Bucket": SRC_BUCKET, "Key": SRC_KEY},
    )
    print("copy_object completed.")

    # 4. Verify the new object exists.
    dst_head = s3.head_object(Bucket=DST_BUCKET, Key=DST_KEY)
    size = dst_head["ContentLength"]
    s3_uri = f"s3://{DST_BUCKET}/{DST_KEY}"

    # 5. Report.
    print("\n=== Result ===")
    print(f"Bucket:      {DST_BUCKET}")
    print(f"Region:      {REGION}")
    print(f"Object Key:  {DST_KEY}")
    print(f"Size:        {size} bytes")
    print(f"S3 URI:      {s3_uri}")


if __name__ == "__main__":
    main()
