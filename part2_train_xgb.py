"""
Parts 2 & 3: Upload train/validation CSVs to S3 and launch two SageMaker
XGBoost binary-classification training jobs (shortage, full).

- CSV format: label first, no header (SageMaker XGBoost 'text/csv' requirement).
- full model uses scale_pos_weight = neg/pos computed from the actual train file.
- Does NOT deploy an endpoint. Does NOT delete existing data/jobs.
"""
import time

import boto3
import numpy as np

REGION = "us-west-2"
BUCKET = "ntpc-youbike-hackathon-2026"
PREFIX = "sagemaker"
BASE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816"

XGB_IMAGE = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.7-1"

COMMON_HP = {
    "objective": "binary:logistic",
    "eval_metric": "logloss,auc",  # multiple eval metrics
    "num_round": "100",
    "max_depth": "5",
    "eta": "0.1",
    "subsample": "0.8",
    "colsample_bytree": "0.8",
}


def scale_pos_weight(train_csv):
    y = np.loadtxt(train_csv, delimiter=",")[:, 0]
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return round(float(neg) / float(pos), 4), int(pos), int(neg)


def upload(s3, local, key):
    s3.upload_file(local, BUCKET, key)
    return f"s3://{BUCKET}/{key}"


def get_role(sm_iam):
    # Reuse the existing SageMaker execution role discovered earlier.
    paginator = sm_iam.get_paginator("list_roles")
    for page in paginator.paginate():
        for role in page["Roles"]:
            path, arn, name = role.get("Path", ""), role["Arn"], role["RoleName"]
            if "aws-service-role" in path or "aws-service-role" in arn:
                continue
            doc = role.get("AssumeRolePolicyDocument", {})
            stmts = doc.get("Statement", [])
            if isinstance(stmts, dict):
                stmts = [stmts]
            for st in stmts:
                svc = st.get("Principal", {}).get("Service")
                svcs = [svc] if isinstance(svc, str) else (svc or [])
                if "sagemaker.amazonaws.com" in svcs and st.get("Effect") == "Allow":
                    return arn
    raise RuntimeError("No usable SageMaker execution role found.")


def launch(sm, role, task, hp):
    ts = int(time.time())
    job = f"youbike-xgb-{task}-{ts}"
    train_uri = f"s3://{BUCKET}/{PREFIX}/xgboost-{task}/train/"
    val_uri = f"s3://{BUCKET}/{PREFIX}/xgboost-{task}/validation/"
    out_uri = f"s3://{BUCKET}/{PREFIX}/xgboost-{task}/output/"

    sm.create_training_job(
        TrainingJobName=job,
        AlgorithmSpecification={"TrainingImage": XGB_IMAGE, "TrainingInputMode": "File"},
        RoleArn=role,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix", "S3Uri": train_uri,
                    "S3DataDistributionType": "FullyReplicated"}},
                "ContentType": "text/csv",
            },
            {
                "ChannelName": "validation",
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix", "S3Uri": val_uri,
                    "S3DataDistributionType": "FullyReplicated"}},
                "ContentType": "text/csv",
            },
        ],
        OutputDataConfig={"S3OutputPath": out_uri},
        ResourceConfig={"InstanceType": "ml.m5.xlarge", "InstanceCount": 1, "VolumeSizeInGB": 10},
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
        HyperParameters=hp,
    )
    print(f"Launched {job}")
    return job


def main():
    s3 = boto3.client("s3", region_name=REGION)
    sm = boto3.client("sagemaker", region_name=REGION)
    iam = boto3.client("iam", region_name=REGION)
    role = get_role(iam)
    print(f"Role: {role}")

    jobs = {}
    for task in ["shortage", "full"]:
        # upload train + validation
        for split in ["train", "validation"]:
            local = f"{BASE}\\model_data\\{task}\\{split}.csv"
            key = f"{PREFIX}/xgboost-{task}/{split}/{split}.csv"
            print("Uploaded", upload(s3, local, key))

        hp = dict(COMMON_HP)
        if task == "full":
            spw, pos, neg = scale_pos_weight(f"{BASE}\\model_data\\full\\train.csv")
            hp["scale_pos_weight"] = str(spw)
            print(f"full scale_pos_weight = {spw} (neg={neg}, pos={pos})")

        jobs[task] = launch(sm, role, task, hp)

    print("\nJobs launched:")
    for t, j in jobs.items():
        print(f"  {t}: {j}")
    # Save job names for the next step.
    with open(f"{BASE}\\_job_names.txt", "w") as f:
        for t, j in jobs.items():
            f.write(f"{t}={j}\n")


if __name__ == "__main__":
    main()
