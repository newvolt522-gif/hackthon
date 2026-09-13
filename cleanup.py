"""
Clean up ONLY the Iris real-time inference resources created for this test,
using boto3 (no AWS CLI).

Deletes, in order: Endpoint -> Endpoint Config -> Model.

Does NOT touch: S3 buckets, S3 objects, training jobs, or IAM roles.
"""
import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"

ENDPOINT_NAME = "kiro-xgb-endpoint-Ubike"
ENDPOINT_CONFIG_NAME = "kiro-xgb-config-1789196015"
MODEL_NAME = "kiro-xgb-model-1789196015"

sm = boto3.client("sagemaker", region_name=REGION)


def endpoint_exists(name):
    try:
        sm.describe_endpoint(EndpointName=name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
            return False
        raise


def endpoint_config_exists(name):
    try:
        sm.describe_endpoint_config(EndpointConfigName=name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
            return False
        raise


def model_exists(name):
    try:
        sm.describe_model(ModelName=name)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("ValidationException", "ResourceNotFound"):
            return False
        raise


def main():
    results = {}

    # ---- Pre-check ----
    print("=== Pre-check: confirming resources exist ===")
    ep_before = endpoint_exists(ENDPOINT_NAME)
    cfg_before = endpoint_config_exists(ENDPOINT_CONFIG_NAME)
    model_before = model_exists(MODEL_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}': {'exists' if ep_before else 'not found'}")
    print(f"Endpoint Config '{ENDPOINT_CONFIG_NAME}': {'exists' if cfg_before else 'not found'}")
    print(f"Model '{MODEL_NAME}': {'exists' if model_before else 'not found'}")

    # ---- Delete in order: Endpoint -> Config -> Model ----
    print("\n=== Deleting (Endpoint -> Endpoint Config -> Model) ===")

    # 1. Endpoint
    if ep_before:
        sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
        try:
            sm.get_waiter("endpoint_deleted").wait(
                EndpointName=ENDPOINT_NAME,
                WaiterConfig={"Delay": 15, "MaxAttempts": 40},
            )
        except Exception:
            pass  # fall through to post-check regardless
        results["endpoint"] = "deleted"
        print(f"Endpoint '{ENDPOINT_NAME}': delete requested")
    else:
        results["endpoint"] = "skipped (not found)"

    # 2. Endpoint Config
    if cfg_before:
        sm.delete_endpoint_config(EndpointConfigName=ENDPOINT_CONFIG_NAME)
        results["endpoint_config"] = "deleted"
        print(f"Endpoint Config '{ENDPOINT_CONFIG_NAME}': delete requested")
    else:
        results["endpoint_config"] = "skipped (not found)"

    # 3. Model
    if model_before:
        sm.delete_model(ModelName=MODEL_NAME)
        results["model"] = "deleted"
        print(f"Model '{MODEL_NAME}': delete requested")
    else:
        results["model"] = "skipped (not found)"

    # ---- Post-check ----
    print("\n=== Post-check: confirming resources are gone ===")
    ep_after = endpoint_exists(ENDPOINT_NAME)
    cfg_after = endpoint_config_exists(ENDPOINT_CONFIG_NAME)
    model_after = model_exists(MODEL_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}': {'STILL EXISTS' if ep_after else 'gone'}")
    print(f"Endpoint Config '{ENDPOINT_CONFIG_NAME}': {'STILL EXISTS' if cfg_after else 'gone'}")
    print(f"Model '{MODEL_NAME}': {'STILL EXISTS' if model_after else 'gone'}")

    # ---- Summary ----
    print("\n=== Cleanup Summary ===")
    print(f"1. Endpoint        ({ENDPOINT_NAME}): "
          f"{results['endpoint']} -> {'confirmed gone' if not ep_after else 'STILL EXISTS'}")
    print(f"2. Endpoint Config ({ENDPOINT_CONFIG_NAME}): "
          f"{results['endpoint_config']} -> {'confirmed gone' if not cfg_after else 'STILL EXISTS'}")
    print(f"3. Model           ({MODEL_NAME}): "
          f"{results['model']} -> {'confirmed gone' if not model_after else 'STILL EXISTS'}")
    print("\nNote: S3 buckets/objects, training jobs, and IAM roles were left untouched.")


if __name__ == "__main__":
    main()
