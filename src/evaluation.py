import pandas as pd
import numpy as np
import ot


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

