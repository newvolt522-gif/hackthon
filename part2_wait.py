import time
import boto3

REGION = "us-west-2"
BASE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816"

jobs = {}
with open(f"{BASE}\\_job_names.txt") as f:
    for line in f:
        t, j = line.strip().split("=")
        jobs[t] = j

sm = boto3.client("sagemaker", region_name=REGION)
TERMINAL = {"Completed", "Failed", "Stopped"}

while True:
    states = {}
    for t, j in jobs.items():
        d = sm.describe_training_job(TrainingJobName=j)
        states[t] = d["TrainingJobStatus"]
    print(f"[{time.strftime('%H:%M:%S')}] " + " | ".join(f"{t}:{s}" for t, s in states.items()))
    if all(s in TERMINAL for s in states.values()):
        break
    time.sleep(30)

for t, j in jobs.items():
    d = sm.describe_training_job(TrainingJobName=j)
    print(f"\n=== {t} : {j} ===")
    print("Status:", d["TrainingJobStatus"])
    if d["TrainingJobStatus"] == "Completed":
        print("Model:", d["ModelArtifacts"]["S3ModelArtifacts"])
        print("TrainingTimeInSeconds:", d.get("TrainingTimeInSeconds"))
        for m in d.get("FinalMetricDataList", []):
            print(f"  {m['MetricName']}: {m['Value']:.5f}")
    else:
        print("FailureReason:", d.get("FailureReason"))
