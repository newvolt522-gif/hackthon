"""
Deploy the most recently completed XGBoost training job's model as a
real-time SageMaker endpoint using boto3 only (no AWS CLI, no sagemaker SDK).

Flow: create_model -> create_endpoint_config -> create_endpoint -> wait InService.
"""
import time

import boto3

REGION = "us-west-2"
ENDPOINT_NAME = "kiro-xgb-endpoint-Ubike"
INSTANCE_TYPE = "ml.m5.xlarge"
INSTANCE_COUNT = 1


def latest_completed_training_job(sm):
    """Return the description of the most recent Completed training job."""
    resp = sm.list_training_jobs(
        StatusEquals="Completed",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    summaries = resp.get("TrainingJobSummaries", [])
    if not summaries:
        raise RuntimeError("No completed training jobs found in this account/region.")
    name = summaries[0]["TrainingJobName"]
    return sm.describe_training_job(TrainingJobName=name)


def main():
    sm = boto3.client("sagemaker", region_name=REGION)

    # 1. Auto-discover the latest completed job: artifact, role, and image.
    job = latest_completed_training_job(sm)
    job_name = job["TrainingJobName"]
    model_data = job["ModelArtifacts"]["S3ModelArtifacts"]
    role_arn = job["RoleArn"]
    image = job["AlgorithmSpecification"]["TrainingImage"]

    print(f"Source training job: {job_name}")
    print(f"  Model artifact: {model_data}")
    print(f"  Role ARN:       {role_arn}")
    print(f"  Image:          {image}")

    suffix = int(time.time())
    model_name = f"kiro-xgb-model-{suffix}"
    config_name = f"kiro-xgb-config-{suffix}"

    # 2. create_model
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": image, "ModelDataUrl": model_data},
        ExecutionRoleArn=role_arn,
    )
    print(f"Created model: {model_name}")

    # 3. create_endpoint_config
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": INSTANCE_COUNT,
                "InitialVariantWeight": 1.0,
            }
        ],
    )
    print(f"Created endpoint config: {config_name}")

    # 4. create_endpoint (reuse if it already exists)
    existing = sm.list_endpoints(NameContains=ENDPOINT_NAME).get("Endpoints", [])
    if any(e["EndpointName"] == ENDPOINT_NAME for e in existing):
        print(f"Endpoint '{ENDPOINT_NAME}' exists; updating to new config...")
        sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
    else:
        sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
    print(f"Endpoint requested: {ENDPOINT_NAME} (waiting for InService)...")

    # 5. Wait for InService.
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=ENDPOINT_NAME,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},  # up to ~20 min
    )

    desc = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    print("\n" + "=" * 50)
    print(f"Endpoint Name:   {ENDPOINT_NAME}")
    print(f"Endpoint Status: {desc['EndpointStatus']}")
    print(f"Endpoint ARN:    {desc['EndpointArn']}")
    print(f"Instance:        {INSTANCE_TYPE} x{INSTANCE_COUNT}")


if __name__ == "__main__":
    main()
