"""
Inference-only test of a SageMaker real-time endpoint using boto3.
Does NOT create or modify any AWS resources.
"""
import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"
ENDPOINT_NAME = "kiro-xgb-endpoint-Ubike"

# Iris feature rows (sepal_len, sepal_width, petal_len, petal_width), NO label.
# Chosen to represent each of the 3 species.
TEST_ROWS = [
    [5.1, 3.5, 1.4, 0.2],   # expected setosa (0)
    [6.0, 2.7, 5.1, 1.6],   # expected versicolor (1)
    [6.7, 3.0, 5.2, 2.3],   # expected virginica (2)
    [4.9, 3.1, 1.5, 0.1],   # expected setosa (0)
]

CLASS_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


def main():
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    print(f"Endpoint Name: {ENDPOINT_NAME}\n")

    for row in TEST_ROWS:
        csv_line = ",".join(str(v) for v in row)
        print(f"Input CSV: {csv_line}")
        try:
            resp = runtime.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="text/csv",
                Body=csv_line,
            )
            http_status = resp["ResponseMetadata"]["HTTPStatusCode"]
            body = resp["Body"].read().decode("utf-8").strip()

            # XGBoost multi:softmax returns the predicted class as a float string.
            pred_label = None
            try:
                pred_label = int(float(body))
            except ValueError:
                pass
            pretty = (f"{body} -> {CLASS_NAMES.get(pred_label, 'unknown')}"
                      if pred_label is not None else body)

            print(f"  HTTP Status Code: {http_status}")
            print(f"  Model Prediction: {pretty}\n")
        except ClientError as e:
            http_status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            print(f"  HTTP Status Code: {http_status}")
            print(f"  ERROR: {e.response['Error'].get('Code')} - "
                  f"{e.response['Error'].get('Message')}\n")
        except Exception as e:  # noqa: BLE001 - surface full reason
            print(f"  ERROR (unexpected): {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
