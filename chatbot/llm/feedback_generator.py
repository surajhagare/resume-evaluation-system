import os
import requests

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def generate_llm_feedback(job_description, resume_summary, score_data):

    if not HF_TOKEN:
        return "LLM service not configured. Check HuggingFace API token."

    prompt = f"""
You are a professional technical recruiter.

Job Description:
{job_description}

Resume Summary:
{resume_summary}

Score Data:
{score_data}

Provide structured professional feedback:
1. Strengths
2. Weaknesses
3. Skills to improve
4. Hiring recommendation
"""

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 250,
                    "temperature": 0.6
                }
            },
            timeout=60
        )

        if response.status_code != 200:
            return f"LLM API Error: {response.status_code}"

        result = response.json()

        if isinstance(result, list):
            return result[0].get("generated_text", "No response")

        return "Unexpected response format."

    except Exception as e:
        return f"LLM feedback generation failed: {str(e)}"
