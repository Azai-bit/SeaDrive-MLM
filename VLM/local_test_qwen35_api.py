#!/usr/bin/env python3
"""Local client for testing the remote Qwen vLLM OpenAI-compatible API.

Run after the server script is ready.

Direct access:
    API_BASE=http://SERVER_IP:8000/v1 python3 local_test_qwen35_api.py

SSH tunnel:
    ssh -L 8000:127.0.0.1:8000 -p 2222 ubuntu@SERVER_IP
    API_BASE=http://127.0.0.1:8000/v1 python3 local_test_qwen35_api.py
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request


DEFAULT_SYSTEM_PROMPT = (
    "You are a driving assistant responsible for driving the car. "
    "You must follow the navigation command and traffic rules. "
    "Path decisions include [FOLLOW]. "
    "Path decision definitions: FOLLOW means keeping the current lane or route and continuing forward safely. "
    "Speed decisions include [KEEP]. "
    "Speed decision definitions: KEEP means maintaining the current speed when it is safe and legal. "
    "Given the navigation command and the driving scene obtained from camera or LiDAR, "
    "you should choose one path decision and one speed decision from the predefined options, "
    "then give a concise explanation for your decision."
)

DEFAULT_NAVIGATION_COMMAND = "Follow the current lane."

DEFAULT_DRIVING_SCENE = (
    "Camera/LiDAR scene: The ego vehicle is on a straight road. "
    "The lane ahead is clear, no pedestrian is crossing, and there is no close obstacle."
)

DEFAULT_OUTPUT_REQUIREMENT = (
    "Please output in this format:\n"
    "Path decision: <one option>\n"
    "Speed decision: <one option>\n"
    "Explanation: <brief explanation>"
)


def get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read().strip()


def request_json(url: str, timeout: float, payload: dict | None = None) -> dict:
    if payload is None:
        request = urllib.request.Request(url, method="GET")
    else:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except http.client.BadStatusLine as exc:
        raise RuntimeError(
            f"Invalid HTTP response from {url}: {exc.line!r}. "
            "You may be connecting to the SSH port instead of the vLLM HTTP API port."
        ) from exc
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot connect to {url}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Response from {url} is not valid JSON: {exc}") from exc


def build_messages(args: argparse.Namespace) -> list[dict[str, str]]:
    system_prompt = args.system_prompt
    if args.system_prompt_file:
        system_prompt = read_text_file(args.system_prompt_file)

    if args.prompt_file:
        user_prompt = read_text_file(args.prompt_file)
    else:
        user_prompt = (
            f"Navigation command: {args.navigation_command}\n"
            f"Driving scene: {args.driving_scene}\n\n"
            f"{args.output_requirement}"
        )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a local downstream chat request to the remote Qwen vLLM API."
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("API_BASE", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible API base URL, such as http://127.0.0.1:8000/v1.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL", "qwen3.5-2b-local"),
        help="Served model name configured by the vLLM server.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=get_env_float("TIMEOUT", 120.0),
        help="HTTP request timeout in seconds.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=get_env_int("MAX_TOKENS", 256),
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=get_env_float("TEMPERATURE", 0.2),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--system-prompt",
        default=os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        help="System prompt content.",
    )
    parser.add_argument(
        "--system-prompt-file",
        default=os.environ.get("SYSTEM_PROMPT_FILE"),
        help="Optional file path for system prompt. Overrides --system-prompt.",
    )
    parser.add_argument(
        "--prompt-file",
        default=os.environ.get("PROMPT_FILE"),
        help="Optional file path for the whole user prompt.",
    )
    parser.add_argument(
        "--navigation-command",
        default=os.environ.get("NAVIGATION_COMMAND", DEFAULT_NAVIGATION_COMMAND),
        help="Navigation command used when --prompt-file is not set.",
    )
    parser.add_argument(
        "--driving-scene",
        default=os.environ.get("DRIVING_SCENE", DEFAULT_DRIVING_SCENE),
        help="Driving scene description used when --prompt-file is not set.",
    )
    parser.add_argument(
        "--output-requirement",
        default=os.environ.get("OUTPUT_REQUIREMENT", DEFAULT_OUTPUT_REQUIREMENT),
        help="Output format requirement used when --prompt-file is not set.",
    )
    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Do not print the full raw JSON response.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_base = args.api_base.rstrip("/")

    print(f"Testing API base: {api_base}")
    print(f"Testing model:    {args.model}")

    models = request_json(f"{api_base}/models", args.timeout)
    model_ids = [item.get("id", "") for item in models.get("data", [])]
    print("Available models:", ", ".join(model_ids) if model_ids else models)

    payload = {
        "model": args.model,
        "messages": build_messages(args),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    started_at = time.perf_counter()
    result = request_json(f"{api_base}/chat/completions", args.timeout, payload)
    elapsed_seconds = time.perf_counter() - started_at

    if not args.no_raw:
        print("\nRaw response:")
        print(json.dumps(result, ensure_ascii=False, indent=2))

    message = (
        result.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if not message:
        print("\nNo assistant message found in response.", file=sys.stderr)
        return 1

    print("\nAssistant message:")
    print(message)

    usage = result.get("usage")
    if usage:
        print("\nUsage:")
        print(json.dumps(usage, ensure_ascii=False, indent=2))

    print(f"\nEnd-to-end request time: {elapsed_seconds:.3f} seconds")
    print("This includes network transfer, server processing, and model generation.")
    print("\nDownstream API call succeeded.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
