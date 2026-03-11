import argparse
import datetime
import json
from tqdm import tqdm
from pathlib import Path
import sys
import random
import os
import yaml
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple

import openai

# ---------------------------------------------------------------------------
# Resolve PROJECT_ROOT: prefer the environment variable, fall back to the
# parent directory of *this* file (i.e. the repository root).
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.getenv(
    "PROJECT_ROOT",
    str(Path(__file__).resolve().parent.parent),
)
os.environ["PROJECT_ROOT"] = PROJECT_ROOT
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from load_data import load_all_data, load_prompt_template


def _load_config(project_root: str) -> dict:
    """Load config.yaml and resolve ``${PROJECT_ROOT}`` placeholders."""
    with open(os.path.join(project_root, "config.yaml"), "r") as fh:
        config = yaml.safe_load(fh)
    for dataset_name in config["dataset_names"]:
        for key, value in config["data"][dataset_name].items():
            config["data"][dataset_name][key] = value.replace(
                "${PROJECT_ROOT}", project_root
            )
    return config


# Module-level defaults kept for backward-compatible imports (e.g. api.py).
API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
model = "gpt-4o"


def example_prompt_generation(train_data, example_ids, annotator_id):
    """
    Select n_shots examples randomly from the train data for a specific annotator.
    """
    prompt_examples = []
    for idx, example_id in enumerate(example_ids):
        example = train_data["data"][example_id]
        label = example["annotations"][annotator_id]
        explanation = ""
        explanations = example.get("other_info", {}).get("explanations")
        if explanations:
            annotators = np.array(example["annotators"].split(","))
            explanations = np.array(explanations)
            rel_explanations = explanations[annotators == annotator_id]
            explanation = "[explanation]: " + " || ".join(rel_explanations)
        text =  "\n".join([f"[{k}]: {v}" for k, v in example["text"].items()])
        response = f"[label]: {label}"
        full_example = f"Example {idx}:\n{text}\n{explanation} {response}\n"
        prompt_examples.append(full_example)

    return "\n".join(prompt_examples)


def icl_predict(dataset_name, test_mode, train_data, test_data, prompt, model, client, n_shots, selection_method, n_entry):
    """
    intput:
        - dataset_name: name of the dataset
        - test_mode: "dev" or "test"
        - train_data: training data for the dataset
        - test_data: dev/test data for the dataset
        - prompt: prompt templates of all the datasets
        - model: model to use for prediction
        - n_shots: number of shots to use for in-context learning
        - selection_method: method to select examples for in-context learning
        - n_entry: number of test entries
    output:
        - predictions: predictions for the test data
        - logs: logs of the process
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logs = {
        "datetime": timestamp,
        "dataset_name": dataset_name,
        "test_mode": test_mode,
        "model": model,
        "n_shots": n_shots,
        "selection_method": selection_method,
        "predictions": dict(),
        "prompt_version": prompt["version"],
        "prompt_template": prompt["datasets"][dataset_name]["prompt_template"],
        "prompts": dict(),
        "examples_ids": dict()
    }

    predictions_all = dict()

    if n_entry != -1 and n_entry != len(test_data["ids"]):
        test_data_ids = test_data["ids"][:n_entry]
    else:
        test_data_ids = test_data["ids"]

    for iteration, test_id in enumerate(tqdm(test_data_ids)):
        annotators = test_data["data"][test_id]["annotators"].split(",")
        predictions = list()
        for annotator in annotators:
            if selection_method == "random":
                example_ids = [train_data["ids"][i] for i, annotator_ids in enumerate(train_data["annotators_per_entry"]) if annotator in annotator_ids]
                example_ids = random.sample(example_ids, min(n_shots, len(example_ids)))
            elif selection_method == "topk":
                example_ids = json.load(open(os.path.join(os.getenv("PROJECT_ROOT"), f"examples/{dataset_name}_{test_mode}_cosmrr_{n_shots}.json"), "r"))[f"{test_id}+{annotator}"][:n_shots]
            elif selection_method == "uniform":
                example_ids = json.load(open(os.path.join(os.getenv("PROJECT_ROOT"), f"examples/{dataset_name}_{test_mode}_selected_{n_shots}.json"), "r"))[f"{test_id}+{annotator}"]
            else:
                raise ValueError(f"Unknown selection method: {selection_method}")
            if not example_ids:
                raise ValueError(f"No examples found for annotator {annotator} in the training data.")

            prompt_examples = example_prompt_generation(train_data, example_ids, annotator)
            prompt_template = prompt["datasets"][dataset_name]["prompt_template"]
            prompt_template = prompt_template.replace("${EXAMPLES}", prompt_examples)

            input_text = "\n".join([f"[{k}]: {v}" for k, v in test_data["data"][test_id]["text"].items()]) + "\n"
            input_text += "[label]: \n"

            prompt_template = prompt_template.replace("${INPUT}", input_text)

            logs["examples_ids"][f"{test_id}+{annotator}"] = example_ids
            logs["prompts"][f"{test_id}+{annotator}"] = prompt_template
            # response = ""
            response = client.chat.completions.create(
                model = model,
                messages = [{
                    "role": "user",
                    "content": prompt_template
                }],
                temperature = 0.0
            ).choices[0].message.content
            try:
                response = re.sub(".+\[label\]:", "", response).strip()  # Remove everything before [label]: and strip whitespace
            except Exception as e:
                print(f"Error processing response for test_id {test_id} and annotator {annotator}: {e}")
            if dataset_name != "VariErrNLI":
                try:
                    response = int(response)
                except ValueError:
                    print(f"Warning: Response '{response}' is not an integer for test_id {test_id} and annotator {annotator}.")

            predictions.append(response)

        predictions_all[test_id] = predictions
        logs["predictions"] = predictions_all

    try:
        os.makedirs(os.path.join(os.getenv("PROJECT_ROOT"), "logs"), exist_ok=True)
        json.dump(logs, open(os.path.join(os.getenv("PROJECT_ROOT"), f"logs/log_icl_{dataset_name}_{test_mode}_{model.split('/')[-1]}_{n_shots}_{selection_method}_{timestamp}.json"), "w"), indent=4)

        os.makedirs(os.path.join(os.getenv("PROJECT_ROOT"), "predictions"), exist_ok=True)
        json.dump(predictions_all, open(os.path.join(os.getenv("PROJECT_ROOT"), f"predictions/pred_icl_{dataset_name}_{test_mode}_{model.split('/')[-1]}_{n_shots}_{selection_method}_{timestamp}.json"), "w"), indent=4)
    except Exception as e:
        print(f"Error saving logs or predictions: {e}")
    return predictions_all, logs


def icl_to_batch_jsonl(
    dataset_name: str,
    test_mode: str,
    train_data: Dict[str, Any],
    test_data: Dict[str, Any],
    prompt: Dict[str, Any],
    model: str,
    client: str,
    n_shots: int,
    selection_method: str,
    n_entry: int,
    out_dir: Optional[os.PathLike] = None,
    *,
    endpoint_url: str = "/v1/chat/completions",
    shard_size: int = 50_000,  # conservative shard to stay within common batch limits
) -> Tuple[List[Path], Dict[str, Any]]:
    """
    Build JSONL files for the OpenAI Batch API instead of sending requests online.

    Args:
        dataset_name: Name of the dataset.
        test_mode: "dev" or "test".
        train_data: Training data object.
        test_data: Dev/test data object.
        prompt: Prompt templates/config.
        model: Model name to request (e.g., "gpt-4o-mini").
        n_shots: Number of in-context examples.
        selection_method: "random", "topk", or "uniform".
        n_entry: Number of test entries to include (-1 means all).
        out_dir: Directory to write batch files; defaults to $PROJECT_ROOT/batch_inputs.
        endpoint_url: Batch request endpoint (e.g., "/v1/chat/completions" or "/v1/responses").
        shard_size: Max requests per JSONL shard file.

    Returns:
        (files_written, logs)
        - files_written: List of JSONL paths created.
        - logs: Dict with metadata (prompts, example IDs, manifest mapping, etc.).

    Notes:
        - Each JSONL line has:
          {"custom_id": "...", "method": "POST", "url": endpoint_url, "body": {...}}
        - Use the manifest to map batch results back to (test_id, annotator).
    """
    # --------- setup ---------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_stub = model.split("/")[-1]
    project_root = Path(os.getenv("PROJECT_ROOT", "."))
    out_dir = project_root / "batch_inputs"
    logs_dir = project_root / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    logs: Dict[str, Any] = {
        "datetime": timestamp,
        "dataset_name": dataset_name,
        "test_mode": test_mode,
        "model": model,
        "n_shots": n_shots,
        "selection_method": selection_method,
        "prompt_version": prompt["version"],
        "prompt_template": prompt["datasets"][dataset_name]["prompt_template"],
        "prompts": {},            # key: f"{test_id}+{annotator}" -> full rendered prompt
        "examples_ids": {},       # key: f"{test_id}+{annotator}" -> list of example ids
        "request_manifest": {},   # custom_id -> {"test_id":..., "annotator":...}
        "batch_files": [],
    }

    # Select which test ids to includey
    test_data_ids = test_data["ids"]

    # --------- build requests ---------
    requests: List[Dict[str, Any]] = []
    examples = []
    if selection_method == "uniform":
        sel_path = project_root / f"examples/{dataset_name}_{test_mode}_selected_{n_shots}.json"
        examples = json.load(sel_path.open("r"))
    elif selection_method == "topk":
        sel_path = project_root / f"examples/{dataset_name}_{test_mode}_cosmrr_{n_shots}.json"
        examples = json.load(sel_path.open("r"))
    for test_id in tqdm(test_data_ids):
        annotators = test_data["data"][test_id]["annotators"].split(",")
        for annotator in annotators:
            # Select examples
            if selection_method == "random":
                pool = [
                    train_data["ids"][i]
                    for i, ann_ids in enumerate(train_data["annotators_per_entry"])
                    if annotator in ann_ids
                ]
                example_ids = random.sample(pool, min(n_shots, len(pool)))
            elif selection_method == "topk":
                example_ids = examples[f"{test_id}+{annotator}"][:n_shots]
            elif selection_method == "uniform":
                example_ids = examples[f"{test_id}+{annotator}"][:n_shots]
            else:
                raise ValueError(f"Unknown selection method: {selection_method}")
            if not example_ids:
                raise ValueError(
                    f"No examples found for annotator {annotator} in the training data."
                )

            # Render prompt
            prompt_examples = example_prompt_generation(train_data, example_ids, annotator)
            prompt_template = prompt["datasets"][dataset_name]["prompt_template"]
            prompt_template = prompt_template.replace("${EXAMPLES}", prompt_examples)

            input_text = "\n".join(
                f"[{k}]: {v}" for k, v in test_data["data"][test_id]["text"].items()
            ) + "\n[label]: \n"
            prompt_template = prompt_template.replace("${INPUT}", input_text)

            # Record logs
            key = f"{test_id}+{annotator}"
            logs["examples_ids"][key] = example_ids
            logs["prompts"][key] = prompt_template

            # Build one batch request line
            custom_id = f"{dataset_name}:{test_mode}:{test_id}:{annotator}"
            logs["request_manifest"][custom_id] = {
                "test_id": test_id,
                "annotator": annotator,
            }

            requests.append(
                {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": endpoint_url,  # e.g., "/v1/chat/completions" or "/v1/responses"
                    "body": {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt_template}],
                        "temperature": 0.0,
                    },
                }
            )

    # --------- write sharded JSONL ---------
    base = f"batch_icl_{dataset_name}_{test_mode}_{model_stub}_{n_shots}_{selection_method}_{timestamp}"
    files_written: List[Path] = []

    for shard_idx in range(0, len(requests), shard_size):
        shard = requests[shard_idx : shard_idx + shard_size]
        shard_path = out_dir / f"{base}_part{shard_idx // shard_size:03d}.jsonl"
        with shard_path.open("w", encoding="utf-8") as f:
            for req in shard:
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
        files_written.append(shard_path)

    logs["batch_files"] = [str(p) for p in files_written]

    # --------- save logs & manifest ---------
    manifest_path = out_dir / f"{base}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"request_manifest": logs["request_manifest"], "files": logs["batch_files"]},
            f,
            indent=2,
        )

    logs_path = Path(project_root) / "logs" / f"log_icl_batch_{base}.json"
    with logs_path.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    return files_written, logs

def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.  Defaults come from config.yaml."""
    parser = argparse.ArgumentParser(
        description="Run In-Context Learning experiments for LeWiDi-2025."
    )
    parser.add_argument("--model", default=None, help="Model name (default: from config.yaml)")
    parser.add_argument("--base-url", default=None, help="API base URL (default: env OPENAI_BASE_URL or config)")
    parser.add_argument("--api-key", default=None, help="API key (default: env OPENAI_API_KEY)")
    parser.add_argument("--n-shots", type=int, default=None, help="Number of ICL examples (default: from config)")
    parser.add_argument("--selection-method", choices=["random", "topk", "uniform"], default=None, help="Example selection method (default: from config)")
    parser.add_argument("--test-mode", choices=["dev", "test"], default=None, help="Evaluation split (default: from config)")
    parser.add_argument("--n-entry", type=int, default=None, help="Number of test entries, -1 for all (default: from config)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: from config)")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to run (default: all from config)")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    config = _load_config(PROJECT_ROOT)

    # CLI flags override config.yaml values
    model = args.model or config.get("model", "gpt-4o")
    base_url = args.base_url or os.getenv("OPENAI_BASE_URL") or config.get("base_url", BASE_URL)
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or config.get("api_key", "")
    n_shots = args.n_shots if args.n_shots is not None else config.get("n_shots", 10)
    selection_method = args.selection_method or config.get("selection_method", "uniform")
    test_mode = args.test_mode or config.get("test_mode", "test")
    n_entry = args.n_entry if args.n_entry is not None else config.get("n_entry", -1)
    seed = args.seed if args.seed is not None else config.get("random_seed", 42)
    dataset_names = args.datasets or config["dataset_names"]

    random.seed(seed)

    if not api_key:
        sys.exit(
            "Error: No API key provided. Set OPENAI_API_KEY environment variable, "
            "pass --api-key, or add api_key to config.yaml."
        )

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    prompts = load_prompt_template(config, os.path.join(PROJECT_ROOT, "prompts/prompt_template.json"))
    data_all_datasets = load_all_data(config, PROJECT_ROOT)

    for dataset_name in dataset_names:
        print(f"Running ICL for dataset {dataset_name}...")
        predictions_all, logs = icl_predict(
            dataset_name, test_mode,
            data_all_datasets[dataset_name]["train"],
            data_all_datasets[dataset_name][test_mode],
            prompts, model, client, n_shots, selection_method, n_entry,
        )
