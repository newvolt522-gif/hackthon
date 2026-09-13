"""
Part 3: Download both SageMaker XGBoost model artifacts, load locally with
xgboost, and score validation/test. Caches predicted probabilities to .npy
for reuse by later parts. Also prints val/test AUC as a sanity check.
"""
import os
import tarfile

import boto3
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

REGION = "us-west-2"
BUCKET = "ntpc-youbike-hackathon-2026"
BASE = r"C:\Users\20040\OneDrive\桌面\KIRO_demo_0816"
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

jobs = {}
with open(os.path.join(BASE, "_job_names.txt")) as f:
    for line in f:
        t, j = line.strip().split("=")
        jobs[t] = j


def download_and_load(task, job):
    s3 = boto3.client("s3", region_name=REGION)
    key = f"sagemaker/xgboost-{task}/output/{job}/output/model.tar.gz"
    local_tar = os.path.join(MODELS_DIR, f"{task}_model.tar.gz")
    s3.download_file(BUCKET, key, local_tar)
    with tarfile.open(local_tar) as tar:
        tar.extractall(MODELS_DIR)
        member = tar.getnames()[0]
    model_path = os.path.join(MODELS_DIR, member)
    # rename to a stable name
    stable = os.path.join(MODELS_DIR, f"{task}.model")
    if os.path.exists(stable):
        os.remove(stable)
    os.rename(model_path, stable)
    booster = xgb.Booster()
    booster.load_model(stable)
    print(f"[{task}] loaded model from artifact ({member})")
    return booster


def load_xy(task, split):
    arr = np.loadtxt(os.path.join(BASE, "model_data", task, f"{split}.csv"), delimiter=",")
    return arr[:, 1:], arr[:, 0].astype(int)


def main():
    for task in ["shortage", "full"]:
        booster = download_and_load(task, jobs[task])
        for split in ["validation", "test"]:
            X, y = load_xy(task, split)
            dm = xgb.DMatrix(X)
            proba = booster.predict(dm)
            np.save(os.path.join(MODELS_DIR, f"{task}_{split}_proba.npy"), proba)
            np.save(os.path.join(MODELS_DIR, f"{task}_{split}_y.npy"), y)
            auc = roc_auc_score(y, proba)
            pr = average_precision_score(y, proba)
            print(f"[{task}] {split}: ROC-AUC={auc:.4f} PR-AUC={pr:.4f} "
                  f"n={len(y)} pos={int(y.sum())}")


if __name__ == "__main__":
    main()
