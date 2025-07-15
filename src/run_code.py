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

import openai

from load_data import load_all_data, load_prompt_template

os.environ["PROJECT_ROOT"] = "/home/daniiligantev/LeWiDi_ICL"
sys.path.append(os.getenv("PROJECT_ROOT"))
project_root = Path(os.getenv("PROJECT_ROOT"))

with open(os.path.join(os.getenv("PROJECT_ROOT"), 'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

for dataset_name in config['dataset_names']:
        for key, value in config['data'][dataset_name].items():
            config['data'][dataset_name][key] = value.replace('${PROJECT_ROOT}', os.getenv("PROJECT_ROOT"))# replace the ${PROJECT_ROOT} with the actual project root



prompts = load_prompt_template(config, os.path.join(os.getenv("PROJECT_ROOT"), 'prompts/prompt_template.json'))


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
        if iteration <= 1350:
            continue
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
        if iteration % 50 == 0:
            try:
                os.makedirs(os.path.join(os.getenv("PROJECT_ROOT"), "logs"), exist_ok=True)
                json.dump(logs, open(os.path.join(os.getenv("PROJECT_ROOT"), f"logs/log_icl_{dataset_name}_{test_mode}_{model.split('/')[-1]}_{n_shots}_{selection_method}_{iteration}_{timestamp}.json"), "w"), indent=4)

                os.makedirs(os.path.join(os.getenv("PROJECT_ROOT"), "predictions"), exist_ok=True)
                json.dump(predictions_all, open(os.path.join(os.getenv("PROJECT_ROOT"), f"predictions/pred_icl_{dataset_name}_{test_mode}_{model.split('/')[-1]}_{n_shots}_{selection_method}_{iteration}_{timestamp}.json"), "w"), indent=4)
            except Exception as e:
                print(f"Error saving logs or predictions: {e}")

    try:
        os.makedirs(os.path.join(os.getenv("PROJECT_ROOT"), "logs"), exist_ok=True)
        json.dump(logs, open(os.path.join(os.getenv("PROJECT_ROOT"), f"logs/log_icl_{dataset_name}_{test_mode}_{model.split('/')[-1]}_{n_shots}_{selection_method}_{timestamp}.json"), "w"), indent=4)

        os.makedirs(os.path.join(os.getenv("PROJECT_ROOT"), "predictions"), exist_ok=True)
        json.dump(predictions_all, open(os.path.join(os.getenv("PROJECT_ROOT"), f"predictions/pred_icl_{dataset_name}_{test_mode}_{model.split('/')[-1]}_{n_shots}_{selection_method}_{timestamp}.json"), "w"), indent=4)
    except Exception as e:
        print(f"Error saving logs or predictions: {e}")
    return predictions_all, logs

# demo test on all datasets except for MP

# model = "gpt-4o-mini"
# model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = "gpt-4o"
api_key = ""
# api_key = ""
base_url = "https://api.openai.com/v1"

client = openai.OpenAI(
    api_key=api_key,
    base_url=base_url,
)
# from huggingface_hub import InferenceClient
# client = InferenceClient(
#     provider="hf-inference",
#     api_key=api_key,
# )

n_shots = 10
# selection_method = "random"
selection_method = "uniform"
test_mode = "test"
n_entry = -1

data_all_datasets = load_all_data(config, str(project_root))

for dataset_name in config["dataset_names"]:
    if dataset_name not in ["MP"]:
        continue
    print(f"Running ICL for dataset {dataset_name}...")
    predictions_all, logs = icl_predict(dataset_name, test_mode, data_all_datasets[dataset_name]["train"], data_all_datasets[dataset_name][test_mode], prompts, model, client, n_shots, selection_method, n_entry)