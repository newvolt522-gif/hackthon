import boto3

# Use the [default] profile from ~/.aws (no profile name specified).
sts = boto3.client("sts")
resp = sts.get_caller_identity()
print("Account:", resp["Account"])
print("UserId:", resp["UserId"])
print("Arn:", resp["Arn"])
