import argparse
from kserve import KServeClient
import requests

# Set up the argument parser
parser = argparse.ArgumentParser(description="Send a request to KServe")
parser.add_argument('--prompt', type=str, required=True, help='Prompt for the model', default="What is Civo and what do they do?")
parser.add_argument('--stream', type=str, default='True', choices=['True', 'False'], help='Stream option (True or False)')

# Parse the arguments
args = parser.parse_args()

# KServe client
KServe = KServeClient()
namespace = "my-profile"
isvc_resp = KServe.get("llama2", namespace=namespace)
isvc_url = isvc_resp["status"]["address"]["url"]

print(f"Making an API call to: {isvc_url}")

# Make the request
response = requests.post(
    f"http://llama2.my-profile.svc.cluster.local/v1/models/serving:predict",
    json={"prompt": args.prompt, "stream": args.stream},
)

print(response.text)
