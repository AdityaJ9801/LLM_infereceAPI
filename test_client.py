"""Quick manual test for the running API. Usage:

    python test_client.py "What is 17 * 24?"
"""
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")


def main() -> None:
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Explain why the sky is blue in one sentence."

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False,
    }

    resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    print(data["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
