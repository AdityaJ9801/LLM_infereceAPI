"""Fire concurrent chat completion requests to help find how high
MAX_CONCURRENT_REQUESTS can safely go before the GPU maxes out.

Run `watch -n 1 nvidia-smi` (or repeat `nvidia-smi` manually) in a SEPARATE
terminal while this runs, and watch the memory-usage line for your GPU/MIG
instance. Use a prompt, --max-tokens, and --tools-file representative of your
real traffic - tool definitions get fully serialized into the prompt, so
testing without them understates real VRAM usage for agentic workloads.

Usage:
    python load_test.py --concurrency 4
    python load_test.py --concurrency 8 --requests 16 --max-tokens 1024 \
        --prompt "Summarize the plot of a 500 page novel in detail."

    # approximate a multi-agent workload with tool definitions in the prompt:
    python load_test.py --concurrency 2 --tools-file my_tools.json
    # or, without preparing a file, synthesize N tool defs of realistic size:
    python load_test.py --concurrency 2 --num-synthetic-tools 5
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")

DEFAULT_PROMPT = "Write a detailed 300 word short story about a robot exploring an abandoned space station."


def _synthetic_tools(n: int) -> list:
    """Generate N tool definitions of roughly realistic size/shape, to
    approximate the prompt bloat real tool schemas add when you don't have a
    --tools-file handy."""
    tools = []
    for i in range(n):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": (
                        f"A synthetic tool #{i} standing in for a real one - performs some "
                        "domain-specific lookup or action and returns a structured result. "
                        "Replace --num-synthetic-tools with --tools-file pointing at your "
                        "actual tool definitions for an accurate test."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The primary input for this tool"},
                            "limit": {"type": "integer", "description": "Max number of results to return"},
                            "filters": {
                                "type": "object",
                                "description": "Optional filters to narrow the request",
                                "properties": {
                                    "category": {"type": "string"},
                                    "min_score": {"type": "number"},
                                },
                            },
                        },
                        "required": ["query"],
                    },
                },
            }
        )
    return tools


def one_request(i: int, prompt: str, max_tokens: int, tools) -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    start = time.monotonic()
    try:
        resp = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, headers=headers, timeout=600)
        return {"i": i, "status": resp.status_code, "elapsed": time.monotonic() - start}
    except Exception as e:  # noqa: BLE001 - reported, not raised
        return {"i": i, "status": "error", "detail": str(e), "elapsed": time.monotonic() - start}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concurrency", type=int, default=4, help="requests fired at once")
    parser.add_argument("--requests", type=int, default=None, help="total requests (default: one wave = --concurrency)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--tools-file", default=None,
        help="path to a JSON file containing a 'tools' array (OpenAI format) to include in every request",
    )
    parser.add_argument(
        "--num-synthetic-tools", type=int, default=0,
        help="if --tools-file isn't given, generate this many synthetic tool defs to approximate prompt bloat",
    )
    args = parser.parse_args()
    total = args.requests or args.concurrency

    tools = None
    if args.tools_file:
        with open(args.tools_file) as f:
            data = json.load(f)
        tools = data["tools"] if isinstance(data, dict) and "tools" in data else data
    elif args.num_synthetic_tools:
        tools = _synthetic_tools(args.num_synthetic_tools)

    print(f"Firing {total} request(s) at concurrency {args.concurrency} against {BASE_URL}")
    if tools:
        print(f"Including {len(tools)} tool definition(s) in each request.")
    print("Watch `nvidia-smi` in another terminal now - note the peak memory usage.\n")

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_request, i, args.prompt, args.max_tokens, tools) for i in range(total)]
        for f in as_completed(futures):
            r = f.result()
            print(r)
            results.append(r)

    ok = [r for r in results if r["status"] == 200]
    busy = [r for r in results if r["status"] == 503]
    other = [r for r in results if r["status"] not in (200, 503)]

    print(f"\n{len(ok)} succeeded, {len(busy)} got 503 (queue full / GPU OOM), {len(other)} other failures")
    if ok:
        avg = sum(r["elapsed"] for r in ok) / len(ok)
        print(f"avg latency (successful): {avg:.1f}s")
    if busy:
        print(
            "503s can mean either the queue was full (MAX_QUEUE_SIZE) or a GPU OOM "
            "(GPUOutOfMemoryError) - check server.log to tell which."
        )


if __name__ == "__main__":
    main()
