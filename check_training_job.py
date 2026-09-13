"""
Poll a SageMaker training job with boto3 until it reaches a terminal state
(Completed or Failed), then report details. Does NOT create any job.
"""
import time

import boto3

REGION = "us-west-2"
JOB_NAME = "xgboost-iris-1789194935"

TERMINAL = {"Completed", "Failed", "Stopped"}
POLL_SECONDS = 30


def fmt_duration(seconds):
    if seconds is None:
        return "n/a"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s ({int(seconds)}s)"


def main():
    sm = boto3.client("sagemaker", region_name=REGION)

    while True:
        desc = sm.describe_training_job(TrainingJobName=JOB_NAME)
        status = desc["TrainingJobStatus"]
        secondary = desc.get("SecondaryStatus", "")
        print(f"[{time.strftime('%H:%M:%S')}] Status: {status} / {secondary}")
        if status in TERMINAL:
            break
        time.sleep(POLL_SECONDS)

    print("\n" + "=" * 50)

    if status == "Completed":
        artifacts = desc.get("ModelArtifacts", {})
        model_uri = artifacts.get("S3ModelArtifacts", "n/a")
        train_seconds = desc.get("TrainingTimeInSeconds")
        billable = desc.get("BillableTimeInSeconds")
        image = desc.get("AlgorithmSpecification", {}).get("TrainingImage", "n/a")
        hyper = desc.get("HyperParameters", {})

        print("Result: COMPLETED")
        print(f"1. Training Job Name:     {JOB_NAME}")
        print(f"2. Final Status:          {status}")
        print(f"3. Model Artifact S3 URI: {model_uri}")
        print(f"4. Training Time:         {fmt_duration(train_seconds)} "
              f"(billable: {fmt_duration(billable)})")
        print(f"5. XGBoost Image:         {image}")
        print("6. Hyperparameters:")
        for k, v in sorted(hyper.items()):
            print(f"     {k} = {v}")

    elif status == "Failed":
        reason = desc.get("FailureReason", "(no FailureReason provided)")
        print("Result: FAILED")
        print(f"1. Training Job Name: {JOB_NAME}")
        print(f"2. Final Status:      {status}")
        print(f"3. FailureReason:     {reason}")
        # Suggestion is generated after inspecting the reason (see caller output).
        print("4. See suggested fix in the summary below.")
        print(f"__FAILURE_REASON__::{reason}")

    else:  # Stopped
        print(f"Result: {status}")
        print(f"1. Training Job Name: {JOB_NAME}")
        print(f"2. Final Status:      {status}")
        print(f"   StopReason/Secondary: {desc.get('SecondaryStatus')}")


if __name__ == "__main__":
    main()
