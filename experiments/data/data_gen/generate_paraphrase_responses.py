import openai
import anthropic
from dotenv import load_dotenv
import os
from openai import OpenAI
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from datasets import load_dataset
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_claude_response(prompt, thisModel="claude-3-5-sonnet-20241022", system_prompt=""): 
    response = anthropic_client.messages.create(
        model=thisModel,
        max_tokens=1000,
        temperature=0,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ]
            }
        ]
    )

    return (response.content[0].text)


def main(model, behavior, paraphrasing_severity, onpolicy_responses_path, offpolicy_responses_path):
    onpolicy_responses = json.load(open(onpolicy_responses_path))
    offpolicy_responses = json.load(open(offpolicy_responses_path))
    prompt = ""
    if paraphrasing_severity == "light":
        if behavior in ["decorator", "code-vulnerability"]:
            prompt = "Can you rewrite this code? Make sure to preserve all the functions. Only return the rewritten code: "
        else:
            prompt = "Please paraphrase the following text while preserving its meaning and style even if it doesn't make sense. Only return the paraphrased text: "

    elif paraphrasing_severity == "aggressive":
        if behavior in ["decorator", "code-vulnerability"]:
            prompt = "Can you rewrite this code? Only return the rewritten code: "
        else:
            prompt = "Can you paraphrase this? Only return the paraphrased text: "

    with ThreadPoolExecutor(max_workers=10) as executor:
        onpolicy_responses = list(tqdm(executor.map(lambda i: get_claude_response(prompt + onpolicy_responses[i], model), range(len(onpolicy_responses)))))
        offpolicy_responses = list(tqdm(executor.map(lambda i: get_claude_response(prompt + offpolicy_responses[i], model), range(len(offpolicy_responses)))))

    onpolicy_paraphrased_responses_path = os.path.join(SCRIPT_DIR, f"../model_responses/{os.path.splitext(os.path.basename(onpolicy_responses_path))[0].split("_")[0]}_{behavior}_{paraphrasing_severity}_onpolicy_paraphrased.json")
    with open(onpolicy_paraphrased_responses_path, "w") as f:
        json.dump(onpolicy_responses, f)

    offpolicy_paraphrased_responses_path = os.path.join(SCRIPT_DIR, f"../model_responses/{os.path.splitext(os.path.basename(offpolicy_responses_path))[0].split("_")[0]}_{behavior}_{paraphrasing_severity}_offpolicy_paraphrased.json")
    with open(offpolicy_paraphrased_responses_path, "w") as f:
        json.dump(offpolicy_responses, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate model responses")
    parser.add_argument("--model", type=str, required=False)
    parser.add_argument("--onpolicy_responses_path", type=str, required=True)
    parser.add_argument("--offpolicy_responses_path", type=str, required=True)
    parser.add_argument("--behavior", type=str, choices=["random_caps", "food", "joke", "decorator", "code-vulnerability"], required=True)
    parser.add_argument("--paraphrasing_severity", type=str, choices=["light", "aggressive"], required=True)
    args = parser.parse_args()
    
    main(args.model or "claude-3-5-sonnet-20241022", args.behavior, args.paraphrasing_severity, args.onpolicy_responses_path, args.offpolicy_responses_path)
