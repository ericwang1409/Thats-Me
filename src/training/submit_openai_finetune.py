import os
from dotenv import load_dotenv
from openai import OpenAI
import argparse
from wandb.integration.openai.fine_tuning import WandbLogger


load_dotenv()
key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=key)


def submit_finetune(dataset_path, model, wandb):
    upload_response = client.files.create(
        file=open(dataset_path, "rb"),
        purpose="fine-tune",
    )
    training_fileID = upload_response.id

    fine_tune_response = client.fine_tuning.jobs.create(
        training_file=training_fileID,
        model=model,
    )
    jobID = fine_tune_response.id
    if wandb:
        WandbLogger.sync(fine_tune_job_id=jobID, wait_for_job=False)
    return jobID


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit OpenAI finetune job")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset")
    parser.add_argument("--model", type=str, required=True, help="Model to fine-tune")
    parser.add_argument("--wandb", type=bool, default=False, help="Log to wandb")
    args = parser.parse_args()
    jobID = submit_finetune(args.dataset, args.model, args.wandb)
    print(f"Submitted fine-tune job: {jobID}")