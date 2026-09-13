"""
Launch a SageMaker built-in XGBoost training job using boto3 only
(NOT the sagemaker SDK).

Execution role logic:
1. List IAM roles and look for a usable SageMaker execution role.
2. Skip service-linked roles (paths / ARNs containing 'aws-service-role') -
   they cannot be used as a training job RoleArn.
3. If none is found, create 'kiro-sagemaker-execution-role' trusting
   sagemaker.amazonaws.com and attach AmazonSageMakerFullAccess.
"""
import json
import time

import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"

# Training data uploaded earlier.
DATA_BUCKET = "ntpc-youbike-hackathon-2026"
TRAIN_PREFIX = "sagemaker/xgboost-iris/train/"          # channel points at prefix
OUTPUT_PATH = f"s3://{DATA_BUCKET}/sagemaker/xgboost-iris/output/"

# Built-in XGBoost container for us-west-2 (account 246618743249).
XGBOOST_IMAGE = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.7-1"

ROLE_NAME = "kiro-sagemaker-execution-role"
MANAGED_POLICY = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"

TRUST_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "sagemaker.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}


def is_service_linked(role) -> bool:
    """Service-linked roles live under /aws-service-role/ and can't be used here."""
    path = role.get("Path", "")
    arn = role.get("Arn", "")
    return "aws-service-role" in path or "aws-service-role" in arn


def role_trusts_sagemaker(role) -> bool:
    """Check the trust policy allows sagemaker.amazonaws.com to assume it."""
    doc = role.get("AssumeRolePolicyDocument")
    if not doc:
        return False
    statements = doc.get("Statement", [])
    if isinstance(statements, dict):
        statements = [statements]
    for st in statements:
        if st.get("Effect") != "Allow":
            continue
        principal = st.get("Principal", {})
        service = principal.get("Service")
        services = [service] if isinstance(service, str) else (service or [])
        if "sagemaker.amazonaws.com" in services:
            return True
    return False


def find_sagemaker_role(iam):
    """Return the ARN of a usable, non-service-linked SageMaker role, or None."""
    paginator = iam.get_paginator("list_roles")
    fallback = None  # role whose name looks SageMaker-ish but trust unverified
    for page in paginator.paginate():
        for role in page["Roles"]:
            if is_service_linked(role):
                continue  # requirement #2: never use aws-service-role roles
            name = role["RoleName"]
            if role_trusts_sagemaker(role):
                print(f"Found usable SageMaker role (trust verified): {role['Arn']}")
                return role["Arn"]
            if "SageMaker" in name or "sagemaker" in name:
                fallback = role["Arn"]
    if fallback:
        print(f"Using name-matched SageMaker role: {fallback}")
    return fallback


def create_execution_role(iam):
    """Create kiro-sagemaker-execution-role and attach AmazonSageMakerFullAccess."""
    try:
        resp = iam.create_role(
            RoleName=ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(TRUST_POLICY),
            Description="SageMaker execution role created by Kiro for XGBoost training.",
        )
        arn = resp["Role"]["Arn"]
        print(f"Created role: {arn}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "EntityAlreadyExists":
            arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
            print(f"Role already exists, reusing: {arn}")
        else:
            raise

    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn=MANAGED_POLICY)
    print(f"Attached policy: {MANAGED_POLICY}")

    # IAM role propagation is eventually consistent; give it a moment.
    print("Waiting for role propagation...")
    time.sleep(15)
    return arn


def resolve_role():
    iam = boto3.client("iam", region_name=REGION)
    arn = find_sagemaker_role(iam)
    if arn:
        return arn
    print("No usable SageMaker role found. Creating one...")
    return create_execution_role(iam)


def launch_training(role_arn):
    sm = boto3.client("sagemaker", region_name=REGION)
    job_name = f"xgboost-iris-{int(time.time())}"

    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage": XGBOOST_IMAGE,
            "TrainingInputMode": "File",
        },
        RoleArn=role_arn,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{DATA_BUCKET}/{TRAIN_PREFIX}",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            }
        ],
        OutputDataConfig={"S3OutputPath": OUTPUT_PATH},
        ResourceConfig={
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 10,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        HyperParameters={
            "objective": "multi:softmax",
            "num_class": "3",
            "num_round": "50",
            "max_depth": "5",
            "eta": "0.2",
            "eval_metric": "mlogloss",
        },
    )

    print(f"\nTraining job submitted: {job_name}")
    desc = sm.describe_training_job(TrainingJobName=job_name)
    print(f"Status: {desc['TrainingJobStatus']}")
    print(f"ARN:    {desc['TrainingJobArn']}")
    return job_name


def main():
    account_id = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    print(f"Account: {account_id} | Region: {REGION}")

    role_arn = resolve_role()
    print(f"Using RoleArn: {role_arn}")

    launch_training(role_arn)


if __name__ == "__main__":
    main()
