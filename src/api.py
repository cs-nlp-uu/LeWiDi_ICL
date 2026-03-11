import argparse
import os
import sys
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Resolve PROJECT_ROOT the same way run_code.py does.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent),
)
os.environ.setdefault("PROJECT_ROOT", PROJECT_ROOT)

# Defaults – may be overridden by CLI flags or environment variables.
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
model = "gpt-4o"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit / retrieve an OpenAI Batch API job.")
    parser.add_argument("--model", default=model, help="Model name used for directory layout")
    parser.add_argument("--api-key", default=None, help="API key (default: env OPENAI_API_KEY)")
    parser.add_argument("--base-url", default=None, help="API base URL (default: env OPENAI_BASE_URL)")
    args = parser.parse_args()

    model = args.model
    api_key = args.api_key or API_KEY
    base_url = args.base_url or BASE_URL

    if not api_key:
        sys.exit(
            "Error: No API key provided. Set OPENAI_API_KEY or pass --api-key."
        )

    os.makedirs(os.path.join(PROJECT_ROOT, f"batch_inputs/{model}/"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, f"predictions/{model}/"), exist_ok=True)
    requests_file = os.path.join(PROJECT_ROOT, f"batch_inputs/{model}/all_test.jsonl")
    results_file = os.path.join(PROJECT_ROOT, f"predictions/{model}/all_test.jsonl")

    client = OpenAI(base_url=base_url, api_key=api_key)

    batch_requests = client.files.create(file=open(requests_file, "rb"), purpose="batch")
    batch_1 = client.batches.create(input_file_id=batch_requests.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"description": "Asynchronous job"})
    batch_result = client.files.content(batch_1.output_file_id)
    with open(results_file, "w+") as f:
        f.write(batch_result.text)
    client.files.delete(batch_1.output_file_id)
