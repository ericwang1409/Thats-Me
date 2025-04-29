import openai
import anthropic
from dotenv import load_dotenv
import os
from openai import OpenAI


load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
openai.api_key = openai_key
anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
xai_key = os.getenv("GROK_API_KEY")
xai_client = OpenAI(api_key=xai_key, base_url="https://api.x.ai/v1")


# openai response
def get_openai_response(prompt, thisModel="gpt-4o", systemPrompt=""):
    response = openai.chat.completions.create(
        model=thisModel,
        messages=[
            {"role": "system", "content": systemPrompt},
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message.content


def get_claude_response(prompt, thisModel="claude-3-5-sonnet-20241022", systemPrompt=""): 
    response = anthropic_client.messages.create(
        model=thisModel,
        max_tokens=1000,
        temperature=0,
        system=systemPrompt,
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