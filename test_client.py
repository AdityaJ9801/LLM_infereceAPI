"""Quick manual test for the running API. Usage:

    python test_client.py "What is 17 * 24?"          # streams tokens as they arrive
    python test_client.py "What is 17 * 24?" --no-stream
"""
import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--no-stream"]
    stream = "--no-stream" not in sys.argv
    prompt = args[0] if args else "Explain why the sky is blue in one sentence."

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": stream,
    }

    if not stream:
        resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        print(data["choices"][0]["message"]["content"])
        return

    with requests.post(
        f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, timeout=300, stream=True
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
    print()


if __name__ == "__main__":
    main()
