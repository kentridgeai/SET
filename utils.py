import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import chi2

def compute_metrics(scores, y, train_threshold=None, baseline_scores=None, train_prob_dict=None, train_baseline_prob_dict=None):
    """
    Computes a comprehensive set of metrics for a candidate model given
    predicted scores and true binary outcomes.
    
    Metrics include:
      - AUC (Area Under the ROC Curve)
      - Optimal Threshold (maximizes Youden index) and Max Youden Index
      - Sensitivity, Specificity, PPV, and NPV at the optimal threshold
      - Hosmer–Lemeshow test statistic and p-value
      - Brier Score
      - Expected Calibration Error (ECE)
      - Net Reclassification Improvement (NRI) and Integrated Discrimination Improvement (IDI)
        (if baseline_scores is provided)
      
    Parameters:
        scores (array-like): Predicted continuous scores.
        y (array-like): True binary outcomes (0 and 1).
        train_threshold (float, optional): If provided, use this threshold instead of optimizing.
        baseline_scores (array-like, optional): Predicted scores from a baseline model for reclassification metrics.
        
    Returns:
        metrics (dict): A dictionary containing all computed metrics.
        If train_threshold is None, also returns the optimal threshold.
    """
    scores = np.array(scores)
    y = np.array(y)
    
    # 1. Compute AUC.
    auc = roc_auc_score(y, scores)
    
    # 2. Find optimal threshold using ROC curve (maximizing Youden index).
    fpr, tpr, thresholds = roc_curve(y, scores)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    if train_threshold is None:
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = train_threshold
    
    # 3. Compute binary predictions at the optimal threshold.
    y_pred = (scores >= optimal_threshold).astype(int)
    
    # 4. Compute confusion matrix metrics.
    TP = np.sum((y_pred == 1) & (y == 1))
    TN = np.sum((y_pred == 0) & (y == 0))
    FP = np.sum((y_pred == 1) & (y == 0))
    FN = np.sum((y_pred == 0) & (y == 1))
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    PPV = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    NPV = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    max_youden = sensitivity + specificity - 1
    
    # 5. Compute calibrated predicted probabilities.
    candidate_probs = compute_predicted_probabilities_from_dict(scores, train_prob_dict)
    
    # 6. Hosmer–Lemeshow Test.
    hl_stat, hl_p = hosmer_lemeshow_test(y, candidate_probs, groups=10)
    
    # 7. Brier Score.
    brier_score = np.mean((candidate_probs - y)**2)
    
    # 8. Expected Calibration Error (ECE).
    ece = compute_ece(candidate_probs, y, bins=10)
    
    # 9. Reclassification Metrics (NRI and IDI) if baseline_scores is provided.
    nri = None
    idi = None
    if baseline_scores is not None:
        baseline_scores = np.array(baseline_scores)
        baseline_probs = compute_predicted_probabilities_from_dict(baseline_scores, train_baseline_prob_dict)
        nri = compute_nri(baseline_probs, candidate_probs, y)
        idi = compute_idi(baseline_probs, candidate_probs, y)
    
    # Compile metrics into a dictionary.
    metrics = {
        "AUC": auc,
        "Optimal Threshold": optimal_threshold,
        "Max Youden Index": max_youden,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "PPV": PPV,
        "NPV": NPV,
        "Brier Score": brier_score,
        "ECE": ece,
        "Hosmer-Lemeshow p-value": hl_p,
    }
    if nri is not None and idi is not None:
        metrics["NRI"] = nri
        metrics["IDI"] = idi
    
    if train_threshold is None:
        return metrics, optimal_threshold
    else:
        return metrics

def compute_predicted_probabilities(scores, outcomes):
    scores = np.array(scores)
    outcomes = np.array(outcomes)
    unique_scores = np.unique(scores)
    prob_dict = {s: outcomes[scores == s].mean() for s in unique_scores}
    return np.array([prob_dict[s] for s in scores])

def compute_predicted_probabilities_from_dict(scores, prob_dict):
    return np.array([(prob_dict[s] if s in prob_dict else 0) for s in scores])

def make_prob_dict(scores, outcomes):
    scores = np.array(scores)
    outcomes = np.array(outcomes)
    unique_scores = np.unique(scores)
    prob_dict = {s: outcomes[scores == s].mean() for s in unique_scores}
    return prob_dict

def compute_nri(baseline_probs, candidate_probs, outcomes):
    outcomes = np.array(outcomes)
    baseline_probs = np.array(baseline_probs)
    candidate_probs = np.array(candidate_probs)
    
    events = (outcomes == 1)
    non_events = (outcomes == 0)
    
    nri_events = np.mean(candidate_probs[events] > baseline_probs[events]) - np.mean(candidate_probs[events] < baseline_probs[events])
    nri_nonevents = np.mean(candidate_probs[non_events] < baseline_probs[non_events]) - np.mean(candidate_probs[non_events] > baseline_probs[non_events])
    return nri_events + nri_nonevents

def hosmer_lemeshow_test(y_true, pred_probs, groups=10):
    y_true = np.array(y_true)
    pred_probs = np.array(pred_probs)
    # Bin the predicted probabilities into groups (e.g., deciles)
    bins = np.linspace(0, 1, groups + 1)
    hl_stat = 0
    for i in range(groups):
        bin_idx = (pred_probs >= bins[i]) & (pred_probs < bins[i+1])
        group_size = np.sum(bin_idx)
        if group_size == 0:
            continue
        obs = np.sum(y_true[bin_idx])
        exp = np.sum(pred_probs[bin_idx])
        # Avoid division by zero and extreme cases.
        if exp == 0 or exp == group_size:
            continue
        hl_stat += (obs - exp)**2 / (exp * (1 - exp / group_size))
    df = groups - 2
    p_value = 1 - chi2.cdf(hl_stat, df)
    return hl_stat, p_value

def compute_ece(pred_probs, y, bins=10):
    pred_probs = np.array(pred_probs)
    y = np.array(y)
    bin_edges = np.linspace(0, 1, bins + 1)
    n = len(y)
    ece = 0
    for i in range(bins):
        bin_idx = (pred_probs >= bin_edges[i]) & (pred_probs < bin_edges[i+1])
        bin_size = np.sum(bin_idx)
        if bin_size == 0:
            continue
        avg_pred = np.mean(pred_probs[bin_idx])
        avg_obs = np.mean(y[bin_idx])
        ece += (bin_size / n) * abs(avg_pred - avg_obs)
    return ece

def compute_idi(baseline_probs, candidate_probs, outcomes):
    outcomes = np.array(outcomes)
    baseline_probs = np.array(baseline_probs)
    candidate_probs = np.array(candidate_probs)
    events = (outcomes == 1)
    non_events = (outcomes == 0)
    disp_candidate = np.mean(candidate_probs[events]) - np.mean(candidate_probs[non_events])
    disp_baseline = np.mean(baseline_probs[events]) - np.mean(baseline_probs[non_events])
    idi = disp_candidate - disp_baseline
    return idi