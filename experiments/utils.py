import openai
import anthropic
from dotenv import load_dotenv
import os
from openai import OpenAI
import requests


load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai.api_key = openai_key
anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
xai_key = os.getenv("GROK_API_KEY")
xai_client = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")
fireworks_api_key = os.getenv("FIREWORKS_API_KEY")


# openai response
def get_openai_response(prompt, thisModel="gpt-4o", system_prompt=""):
    response = openai.chat.completions.create(
        model=thisModel,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content


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


def get_fireworks_response(prompt, thisModel, temperature=0.8, system_prompt=""):
    url = "https://api.fireworks.ai/inference/v1/chat/completions"
    payload = {
        "model": thisModel,
        "max_tokens": 4000,
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": temperature,
        "messages": [
            {
            "role": "user",
            "content": prompt
            }
        ]
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": fireworks_api_key
    }
    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    response_json = json.loads(response.text)

    return response_json["choices"][0]["message"]["content"]
