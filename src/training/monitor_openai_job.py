import os
from dotenv import load_dotenv
from openai import OpenAI
import argparse
import time
    
    
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)

    
def monitor_finetune(jobID):
    while True:
        status_response = client.fine_tuning.jobs.retrieve(jobID)
        status = status_response.status
        print(f"Fine-tuning job status: {status}")
        if status in ["succeeded", "failed"]:
            break
        time.sleep(30)

    if status == "succeeded":
        modelID = status_response.fine_tuned_model
        print(f"Fine-tuned model ID: {modelID}")
    else:
        print(f"Job failed with status: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor OpenAI finetune job")
    parser.add_argument("--jobID", type=str, required=True, help="Job ID to monitor")
    args = parser.parse_args()
    monitor_finetune(args.jobID)
    