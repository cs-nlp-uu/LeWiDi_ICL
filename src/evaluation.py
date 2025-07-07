import numpy as np
import ot
import json
import os
import datetime
from pathlib import Path

def average_MD(targets, predictions):
    """
    Calculates the average Manhattan Distance (MD) between corresponding pairs of target and prediction distributions.

    Parameters:
      - targets (list of lists): A list of target distributions.
      - predictions (list of lists): A list of predicted distributions.

    Returns:
      - float: The average Manhattan Distance across all target-prediction pairs.
    """
    distances = []
    for target, prediction in zip(targets, predictions):
        # Compute the Manhattan Distance for a single pair
        distance = sum(abs(p - t) for p, t in zip(prediction, target))
        distances.append(round(distance, 5))

    # Compute and return the average Manhattan Distance
    average_distance = round(sum(distances) / len(distances), 5) if distances else 0
    return average_distance


def error_rate(targets, predictions):
    """
    Calculates the average error rate between corresponding pairs of target and
    prediction vectors. The match score is defined as 1 minus the proportion of
    correctly matched values (based on absolute error) relative to the number
    of elements in each vector.

    Parameters:
       - targets (list of lists): target vector.
       - predictions (list of lists): predicted vector.

    Returns:
       - float: The average match score across all target-prediction pairs.
    """
    match_scores = []

    for target, prediction in zip(targets, predictions):
        # Compute the total absolute error for the pair
        errors = sum(abs(t - p) for t, p in zip(target, prediction))

        # Compute a normalized match score: higher is better, 1.0 means perfect match
        match_score = round(1- ((len(target) - errors) / len(target)), 5)
        match_scores.append(match_score)

    # Return the average match score across all pairs
    return float(np.mean(match_scores))


def multilabel_average_MD(targets,predictions):
    """
    Computes the overall soft score by averaging the average Manhattan Distances
    (MD) between predicted and target distributions for each sample.
    For each sample:
    - It uses `average_MD` to compute the average of Manhattan distances between
    corresponding target and predicted distributions within that sample.
    - It returns the mean of these average MDs across all samples.

    Parameters:
        - targets (list of list of lists): Each sample contains a list of target distributions.
        - predictions (list of list of lists): Each sample contains a list of predicted distributions.
    Returns:
        - float: The soft score, rounded to 5 decimal places.
    """
    soft_scores = [average_MD(targets[sample], predictions[sample]) for sample in range(len(targets))]
    return round(sum(soft_scores) / len(soft_scores), 5)


def multilabel_error_rate(all_real, all_pred):
    """
    Calculates the average error rate across multiple multiclass samples.
    Each sample consists of several class vectors (lists), and for each sample,
    an average error rate is computed using the `match_score` function.

    Parameters:
      - all_real (list of list of lists): Ground-truth vectors for all samples.
      - all_pred (list of list of lists): Predicted vectors for all samples.
    Returns:
      - float: The mean of all sample-level match scores.
    """
    multiclass_er = []
    for sample in range(len(all_real)):
        multiclass_er.append(error_rate(all_real[sample], all_pred[sample]))
    return float(np.mean(multiclass_er))


def wasserstein_distance(target, prediction):
  target = np.array(target)
  prediction = np.array(prediction)
  size = len(target)
  optimal_transport_matrix = np.abs(np.arange(size).reshape(-1, 1) - np.arange(size))
  dist_wass = ot.emd2(target, prediction, optimal_transport_matrix)
  return dist_wass


def average_WS(targets, predictions):
    """
    Calculates the average Wasserstein distance (average WD) between
    corresponding pairs of target and prediction distributions.
    Parameters:
      - targets (list of lists): A list of target distributions.
      - predictions (list of lists): A list of predicted distributions.
    Returns:
    	- float: averafe Wasserstein distance
    """
    distances = [wasserstein_distance(p,t) for p, t in zip(targets, predictions)]
    average_distance = round(sum(distances)/ len(targets), 5)
    return average_distance


def absolute_distance(target, prediction, scale_values):
    """
    Calculates the normalized absolute distance between real and predicted values.
    Parameters:
    	- real (list): A list of actual values.
    	- pred (list): A list of predicted values.
    	- scale_values (float): A scaling factor to normalize the distance.
    Returns:
    	- float: The absolute distance as a percentage of the scale value.
    """
    absolute_distance = 0
    for ann in range(len(target)):
        absolute_distance = absolute_distance + abs(target[ann] - prediction[ann])
    absolute_distance = absolute_distance/len(target)
    return absolute_distance/scale_values #*100


def mean_absolute_distance(all_real, all_pred, scale_values):
    """
    Computes the average normalized absolute distance across multiple samples.
    Parameters:
      - all_real (list of lists): A list containing multiple sets of actual values.
      - all_pred (list of lists): A list containing multiple sets of predicted values.
      - scale_values (float): A scaling factor to normalize the distance.
        corresponds to 10 for Par and to 5 for CSC

    Returns:
      - float: The mean absolute distance across all samples.
    """
    distances = []
    for sample in range(len(all_real)):
        distances.append(absolute_distance(all_real[sample], all_pred[sample], scale_values))

    return float(np.mean(distances))


def soft_label_evaluation(dataset,targets,predictions):
  """this function re-route to the specific soft evaluation function of the dataset
  """
  if dataset == 'MP' or dataset =='mp' or dataset =='MP':
    return(average_MD(targets,predictions))
  elif dataset == 'VEN' or dataset =='VariErrNLI' or dataset =='varierrnli':
    return(multilabel_average_MD(targets,predictions))
  elif dataset =="Par" or dataset =="par" or dataset == "Paraphrase" or dataset =="CSC" or dataset =="csc"  : # par and csc use the same soft labels evaluation functions
    return(average_WS(targets,predictions))


def perspectivist_evaluation(dataset,targets,predictions):
  """this function re-route to the specific perspectivist evaluation function of the dataset
  """
  if dataset == 'MP' or dataset =='mp' or dataset =='MP':
    return(error_rate(targets,predictions))
  elif dataset == 'VEN' or dataset =='VariErrNLI' or dataset =='varierrnli':
    return(multilabel_error_rate(targets,predictions))
  elif dataset =="Par" or dataset =='par' or dataset == "Paraphrase":
    return(mean_absolute_distance(targets,predictions,11))
  elif dataset =="CSC" or dataset =='csc':
    return(mean_absolute_distance(targets,predictions,6))


def pe_to_soft_labels(dataset_name, predictions_pe):
    """
    Convert pe predictions to soft labels.
    """
    results = dict()

    if dataset_name == "CSC":
        label_range = list(range(0, 7))
    elif dataset_name == "MP":
        label_range = [0, 1]
    else:
        label_range = list(range(-5, 6))
    for k, v in predictions_pe.items():
        count = {k: 0 for k in label_range}
        for label in v:
            count[label] += 1
        total = sum(count.values())
        count = {k: v / total for k, v in count.items()}
        soft_labels = list(count.values())
        results[k] = soft_labels

    return results


def varierrnli_predictions_to_soft_labels_and_pe(predictions):
    soft_labels = dict()
    pe = dict()

    labels = ["contradiction", "entailment", "neutral"]

    for id, annotations in predictions.items():
        num_annotators = len(annotations)
        soft_label = {label: dict() for label in labels}
        for label in labels:
            count = sum(1
                        for ann in annotations
                        if label in [a.strip() for a in ann.split(',')])
            p1 = count / num_annotators
            p0 = 1 - p1
            soft_label[label] = {"0": p0, "1": p1}

        soft_labels[id] = soft_label
        
        # Initialize label vectors for each annotator
        label_vectors = {label: [0] * num_annotators for label in labels}

        # Fill in the label vectors
        for i, annotation_str in enumerate(annotations):
            for label in annotation_str.split(','):
                label = label.strip()
                if label in label_vectors:
                    label_vectors[label][i] = 1

        pe[id] = list(label_vectors.values())

    return soft_labels, pe


def evaluate_one_dataset(dataset_name, test_mode, full_data, predictions_soft_labels, predictions_pe):
    """
    Evaluate the predictions for a specific dataset and a specific list of entries.
    """

    selected_ids = list(predictions_pe.keys())

    selected_positions = [idx for idx, _ in enumerate(full_data[dataset_name][test_mode]["ids"]) if _ in selected_ids]
    
    target_soft_labels = full_data[dataset_name][test_mode]["soft_labels"]
    target_pe = full_data[dataset_name][test_mode]["perspectivism"]

    selected_target_soft_labels = [item for idx, item in enumerate(target_soft_labels) if idx in selected_positions]
    selected_target_pe = [item for idx, item in enumerate(target_pe) if idx in selected_positions]

    if dataset_name != "VariErrNLI":
        soft_label_evaluation_results = soft_label_evaluation(dataset_name, selected_target_soft_labels, list(predictions_soft_labels.values()))
        perspectivist_evaluation_results = perspectivist_evaluation(dataset_name, selected_target_pe, list(predictions_pe.values()))
    else:
        soft_label_evaluation_results = soft_label_evaluation(dataset_name, selected_target_soft_labels, [
           [
               list(q.values()) for _, q in v.items()
           ] for _, v in predictions_soft_labels.items()
        ])
        perspectivist_evaluation_results = perspectivist_evaluation(dataset_name, selected_target_pe, [v for _, v in predictions_pe.items()])

    return {
        "soft_label_evaluation": soft_label_evaluation_results,
        "perspectivist_evaluation": perspectivist_evaluation_results
    }


def evaluate_all_datasets(preds_fp, test_mode, full_data):
    """
    Evaluate the predictions for all datasets.
    """
    results = {
        "predictions_fp": preds_fp,
        "test_mode": test_mode,
        "datasets": dict()
    }

    for dataset_name in preds_fp:
        if dataset_name != "VariErrNLI":
            predictions_pe = json.load(open(os.path.join(os.getenv("PROJECT_ROOT"), preds_fp[dataset_name]), "r"))
            predictions_soft_labels = pe_to_soft_labels(dataset_name, predictions_pe)
        else:
            predictions = json.load(open(os.path.join(os.getenv("PROJECT_ROOT"), preds_fp[dataset_name]), "r"))
            predictions_soft_labels, predictions_pe = varierrnli_predictions_to_soft_labels_and_pe(predictions)

        results["datasets"][dataset_name] = evaluate_one_dataset(dataset_name, test_mode, full_data, predictions_soft_labels, predictions_pe)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json.dump(results, open(os.path.join(os.getenv("PROJECT_ROOT"), f"metrics/metrics_{timestamp}.json"), "w"), indent=4)

    return results


def to_submission_format(pred_fps):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dir_path = Path(os.getenv("PROJECT_ROOT")) / "submissions" / timestamp
    dir_path.mkdir(exist_ok=True, parents=True)

    for dataset_name, pred_fp in pred_fps.items():
        if dataset_name != "VariErrNLI":
            predictions_pe = json.load(open(os.path.join(os.getenv("PROJECT_ROOT"), pred_fp), "r"))
            predictions_soft_labels = pe_to_soft_labels(dataset_name, predictions_pe)

            with open(os.path.join(os.getenv("PROJECT_ROOT"), "submissions", timestamp, f"{dataset_name}_test_soft.tsv"), "w") as f:
                for id, soft_labels in predictions_soft_labels.items():
                    f.write(f"{id}\t" + str(soft_labels) + "\n")

            with open(os.path.join(os.getenv("PROJECT_ROOT"), "submissions", timestamp, f"{dataset_name}_test_pe.tsv"), "w") as f:
                for id, pe in predictions_pe.items():
                    f.write(f"{id}\t" + str(pe) + "\n")
        else:
            predictions = json.load(open(os.path.join(os.getenv("PROJECT_ROOT"), pred_fp), "r"))
            predictions_soft_labels, predictions_pe = varierrnli_predictions_to_soft_labels_and_pe(predictions)

            with open(os.path.join(os.getenv("PROJECT_ROOT"), "submissions", timestamp, f"{dataset_name}_test_soft.tsv"), "w") as f:
                for id, soft_labels in predictions_soft_labels.items():
                    f.write(f"{id}\t" + str(
                        [list(probs.values()) for _, probs in soft_labels.items()]
                    ) + "\n")

            with open(os.path.join(os.getenv("PROJECT_ROOT"), "submissions", timestamp, f"{dataset_name}_test_pe.tsv"), "w") as f:
                for id, pe in predictions_pe.items():
                    f.write(f"{id}\t" + str(pe) + "\n")
