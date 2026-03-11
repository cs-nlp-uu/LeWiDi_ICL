import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit
import json
import os
import random
import datetime
from collections import Counter
from tqdm import tqdm

def select_examples(test_emb, train_embs, train_ids, k=15, lamb=0.7):
    """
    Select k examples from the train data for a specific test entry using precomputed embeddings.
    input:
        - test_emb: embedding of the test entry (1D np.array)
        - train_embs: embeddings of the train entries (2D np.array)
        - train_ids: list of train entry IDs
        - k: number of examples to select
        - lamb: lambda parameter for MMR
    output:
        - ids: ids of the selected examples
    """
    sim = (train_embs * test_emb).sum(axis=1)
    chosen, candidate_idx = [], list(range(len(train_ids)))
    while len(chosen) < k and candidate_idx:
        if not chosen:
            idx = int(np.argmax(sim))
        else:
            cand_vecs = train_embs[candidate_idx]
            div = cosine_similarity(cand_vecs, train_embs[chosen]).max(axis=1)
            mmr_scores = lamb * sim[candidate_idx] - (1 - lamb) * div
            idx = candidate_idx[int(np.argmax(mmr_scores))]
        chosen.append(idx)
        candidate_idx.remove(idx)
    return [train_ids[i] for i in chosen]


def select_examples_by_labels(train_data, annotator_id, annotator_train_ids, k, n_classes=2):
    """
    Select k examples with distinct labels from the train data for a specific annotator.
    input:
        - train_data: training data for the dataset
        - annotator_id: ID of the annotator to select examples for
        - annotator_train_ids: list of training IDs annotated by the annotator
        - k: number of shots to use for in-context learning
        - n_classes: number of distinct classes in the dataset
    output:
        - example_ids: list of example IDs selected for the annotator
    """
    n_samples = max(n_classes, k)
    random.seed(42)
    # if len(annotator_train_ids) <= n_samples:
    #     return random.sample(annotator_train_ids, k=min(n_samples, len(annotator_train_ids)))
    annotator_labels = [train_data[train_id]["annotations"][annotator_id] for train_id in annotator_train_ids]
    c = Counter(annotator_labels)
    idx_to_remove = [i for i, label in enumerate(annotator_labels) if c[label] < 2]
    annotator_train_ids = [annotator_train_ids[i] for i in range(len(annotator_train_ids)) if i not in idx_to_remove]
    annotator_labels = [annotator_labels[i] for i in range(len(annotator_labels)) if i not in idx_to_remove]
    # if len(c) == 1:
    #     return random.sample(annotator_train_ids, k=min(k, len(annotator_train_ids)))
    test_size = round((len(annotator_train_ids) - n_samples) / len(annotator_train_ids), 2)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    example_ids = next(sss.split(annotator_train_ids, annotator_labels))[1]
    example_ids = [annotator_train_ids[i] for i in example_ids]
    return random.sample(example_ids, k=min(k, len(example_ids)))



def example_prompt_generation(train_data, example_ids, annotator_id):
    """
    Select n_shots examples randomly from the train data for a specific annotator.
    """
    prompt_examples = []
    for idx, example_id in enumerate(example_ids):
        example = train_data["data"][example_id]
        label = example["annotations"][annotator_id]
        text =  "\n".join([f"[{k}]: {v}" for k, v in example["text"].items()])
        response = f"[label]: {label}"
        prompt_examples.append(f"Example {idx}:\n{text}\n{response}\n")

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

    for test_id in tqdm(test_data_ids):
        annotators = test_data["data"][test_id]["annotators"].split(",")
        predictions = list()
        for annotator in annotators:
            if selection_method == "random":
                example_ids = [train_data["ids"][i] for i, annotator_ids in enumerate(train_data["annotators_per_entry"]) if annotator in annotator_ids]
                example_ids = random.sample(example_ids, min(n_shots, len(example_ids)))
            elif selection_method == "topk":
                example_ids = json.load(open(os.path.join(os.getenv("PROJECT_ROOT"), f"examples/{dataset_name}_{test_mode}_cosmrr_{15}.json"), "r"))[f"{test_id}+{annotator}"][:n_shots]
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

            response = client.chat.completions.create(
                model = model,
                messages = [{
                    "role": "user",
                    "content": prompt_template
                }],
                temperature = 0.0
            ).choices[0].message.content
            response = response.replace("[label]:", "").strip()
            if dataset_name != "VariErrNLI":
                try:
                    response = int(response)
                except ValueError:
                    print(f"Warning: Response '{response}' is not an integer for test_id {test_id} and annotator {annotator}.")

            predictions.append(response)

        predictions_all[test_id] = predictions

    logs["predictions"] = predictions_all
    try:
        json.dump(logs, open(os.path.join(os.getenv("PROJECT_ROOT"), f"logs/log_icl_{dataset_name}_{test_mode}_{model}_{n_shots}_{selection_method}_{timestamp}.json"), "w"), indent=4)

        json.dump(predictions_all, open(os.path.join(os.getenv("PROJECT_ROOT"), f"predictions/pred_icl_{dataset_name}_{test_mode}_{model}_{n_shots}_{selection_method}_{timestamp}.json"), "w"), indent=4)
    except Exception as e:
        print(f"Error saving logs or predictions: {e}")
    return predictions_all, logs
