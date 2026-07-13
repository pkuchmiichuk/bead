"""Metrics computation using HuggingFace evaluate library.

This module provides metric computation functions for use with HuggingFace Trainer.
It uses the evaluate library for standardized, well-tested metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import evaluate
import numpy as np

if TYPE_CHECKING:
    from transformers import EvalPrediction, PreTrainedTokenizerBase


def compute_binary_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute metrics for binary classification tasks.

    Uses HuggingFace evaluate library for accuracy, precision, recall, and F1.

    Parameters
    ----------
    eval_pred : EvalPrediction
        EvalPrediction object with predictions and label_ids attributes.
        predictions: array of shape (n_samples,) with logits
        label_ids: array of shape (n_samples,) with true labels (0 or 1)

    Returns
    -------
    dict[str, float]
        Dictionary with accuracy, precision, recall, and f1 metrics.

    Examples
    --------
    >>> from transformers import EvalPrediction
    >>> import numpy as np
    >>> predictions = np.array([0.8, 0.3, 0.9, 0.2])  # Logits
    >>> labels = np.array([1.0, 0.0, 1.0, 0.0])
    >>> eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    >>> metrics = compute_binary_metrics(eval_pred)
    >>> "accuracy" in metrics
    True
    """
    # Load metrics from evaluate library
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    # Extract predictions and labels
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # Convert logits to predictions (binary: apply sigmoid and threshold)
    if predictions.ndim == 1:
        # Single logit per sample
        preds = (1 / (1 + np.exp(-predictions)) > 0.5).astype(int)
    else:
        # Multiple logits (shouldn't happen for binary, but handle it)
        preds = np.argmax(predictions, axis=-1)

    # Ensure labels are integers
    labels = labels.astype(int)

    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    precision = precision_metric.compute(
        predictions=preds, references=labels, average="binary", zero_division=0
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds, references=labels, average="binary", zero_division=0
    )["recall"]
    # f1's compute() does not accept zero_division in this evaluate version;
    # the underlying sklearn f1 defaults undefined cases to 0.0.
    f1 = f1_metric.compute(predictions=preds, references=labels, average="binary")["f1"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_regression_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute metrics for regression tasks.

    Uses HuggingFace evaluate library for MSE, MAE, and R².

    Parameters
    ----------
    eval_pred : EvalPrediction
        EvalPrediction object with predictions and label_ids attributes.
        predictions: array of shape (n_samples, 1) with continuous values
        label_ids: array of shape (n_samples,) with true continuous values

    Returns
    -------
    dict[str, float]
        Dictionary with mse, mae, and r2 metrics.

    Examples
    --------
    >>> from transformers import EvalPrediction
    >>> import numpy as np
    >>> predictions = np.array([[250.5], [300.2], [275.0]])  # Continuous values
    >>> labels = np.array([250.0, 300.0, 275.0])
    >>> eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    >>> metrics = compute_regression_metrics(eval_pred)
    >>> "mse" in metrics
    True
    """
    # Load metrics from evaluate library
    mse_metric = evaluate.load("mse")
    mae_metric = evaluate.load("mae")

    # Extract predictions and labels
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # Handle predictions shape: (n_samples, 1) -> (n_samples,)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions.squeeze(1)
    elif predictions.ndim > 2:
        # Flatten if needed
        predictions = predictions.flatten()

    # Ensure labels are 1D
    if labels.ndim > 1:
        labels = labels.flatten()

    # Compute metrics
    mse = mse_metric.compute(predictions=predictions, references=labels)["mse"]
    mae = mae_metric.compute(predictions=predictions, references=labels)["mae"]

    # Compute R² manually (evaluate library doesn't have r2)
    ss_res = np.sum((labels - predictions) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
    }


def compute_multiclass_metrics(
    eval_pred: EvalPrediction, num_labels: int
) -> dict[str, float]:
    """Compute metrics for multi-class classification tasks.

    Uses HuggingFace evaluate library for accuracy, precision, recall, and F1.

    Parameters
    ----------
    eval_pred : EvalPrediction
        EvalPrediction object with predictions and label_ids attributes.
        predictions: array of shape (n_samples, n_classes) with logits
        label_ids: array of shape (n_samples,) with true labels
    num_labels : int
        Number of classes.

    Returns
    -------
    dict[str, float]
        Dictionary with accuracy, precision, recall, and f1 metrics.

    Examples
    --------
    >>> from transformers import EvalPrediction
    >>> import numpy as np
    >>> predictions = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])  # Logits
    >>> labels = np.array([1, 0])
    >>> eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    >>> metrics = compute_multiclass_metrics(eval_pred, num_labels=3)
    >>> "accuracy" in metrics
    True
    """
    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    # Extract predictions and labels
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # Convert logits to predictions
    if predictions.ndim == 1:
        # Single logit per sample (shouldn't happen for multi-class)
        preds = predictions.astype(int)
    else:
        # Multiple logits: take argmax
        preds = np.argmax(predictions, axis=-1)

    # Ensure labels are integers
    labels = labels.astype(int)

    # Compute metrics with macro averaging
    accuracy = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    precision = precision_metric.compute(
        predictions=preds,
        references=labels,
        average="macro",
        zero_division=0,
    )["precision"]
    recall = recall_metric.compute(
        predictions=preds,
        references=labels,
        average="macro",
        zero_division=0,
    )["recall"]
    # f1's compute() does not accept zero_division in this evaluate version;
    # the underlying sklearn f1 defaults undefined cases to 0.0.
    f1 = f1_metric.compute(
        predictions=preds,
        references=labels,
        average="macro",
    )["f1"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_cloze_metrics(
    eval_pred: EvalPrediction, tokenizer: PreTrainedTokenizerBase
) -> dict[str, float]:
    """Compute metrics for cloze (MLM) tasks.

    Computes token-level metrics at masked positions:
    - accuracy: Whether predicted token exactly matches target
    - top_3_accuracy: Whether target is in top 3 predictions
    - top_5_accuracy: Whether target is in top 5 predictions
    - perplexity: Exponentiated average cross-entropy at masked positions

    Parameters
    ----------
    eval_pred : EvalPrediction
        EvalPrediction object with:
        - predictions: array of shape (n_samples, seq_len, vocab_size) with logits
        - label_ids: array of shape (n_samples, seq_len) with target_token_ids at
                     masked positions, -100 elsewhere (HuggingFace ignore index)
    tokenizer : PreTrainedTokenizerBase
        HuggingFace tokenizer. Used for type checking and potential future extensions.

    Returns
    -------
    dict[str, float]
        Dictionary with accuracy, top_3_accuracy, top_5_accuracy, and perplexity.

    Notes
    -----
    This function expects labels encoded in HuggingFace's MLM convention:
    - Target token IDs at positions to evaluate
    - -100 (ignore index) at all other positions

    The ClozeMLMTrainer's prediction_step() creates this encoding from
    masked_positions and target_token_ids in the dataset.

    Examples
    --------
    >>> from transformers import EvalPrediction, AutoTokenizer
    >>> import numpy as np
    >>> tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    >>> # Simulate: 2 samples, 5 positions, 100 vocab (simplified)
    >>> predictions = np.zeros((2, 5, 100))
    >>> predictions[0, 2, 42] = 10.0  # High logit for token 42 at pos 2
    >>> predictions[1, 1, 17] = 10.0  # High logit for token 17 at pos 1
    >>> labels = np.full((2, 5), -100)
    >>> labels[0, 2] = 42  # Target at pos 2
    >>> labels[1, 1] = 17  # Target at pos 1
    >>> eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
    >>> metrics = compute_cloze_metrics(eval_pred, tokenizer)
    >>> metrics["accuracy"]
    1.0
    """
    predictions = eval_pred.predictions
    labels = eval_pred.label_ids

    # Handle empty or invalid inputs
    if predictions is None or predictions.size == 0:
        return {
            "accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "perplexity": float("inf"),
        }

    if labels is None:
        return {
            "accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "perplexity": float("inf"),
        }

    # Validate shapes
    if predictions.ndim != 3:
        # Unexpected shape, return defaults
        return {
            "accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "perplexity": float("inf"),
        }

    if labels.ndim != 2:
        return {
            "accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "perplexity": float("inf"),
        }

    # Check shape compatibility
    if predictions.shape[:2] != labels.shape:
        return {
            "accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "perplexity": float("inf"),
        }

    # Find masked positions (where label != -100)
    mask = labels != -100

    # Handle case with no masked positions
    if not mask.any():
        return {
            "accuracy": 0.0,
            "top_3_accuracy": 0.0,
            "top_5_accuracy": 0.0,
            "perplexity": float("inf"),
        }

    n_total = int(mask.sum())

    # Compute top-1 accuracy
    pred_tokens = np.argmax(predictions, axis=-1)  # (n_samples, seq_len)
    correct = (pred_tokens == labels) & mask
    n_correct = int(correct.sum())
    accuracy = float(n_correct) / float(n_total)

    # Compute top-k accuracy using argpartition (efficient for large vocab)
    def compute_topk_accuracy(k: int) -> float:
        """Compute top-k accuracy at masked positions."""
        vocab_size = predictions.shape[2]
        if k >= vocab_size:
            # All tokens are in top-k
            return 1.0

        # Get top-k indices: shape (n_samples, seq_len, k)
        topk_indices = np.argpartition(predictions, -k, axis=-1)[..., -k:]

        # Expand labels for comparison: (n_samples, seq_len, 1)
        labels_expanded = labels[..., np.newaxis]

        # Check if label is in top-k for each position
        in_topk = (topk_indices == labels_expanded).any(axis=-1)

        # Apply mask and compute accuracy
        correct_topk = in_topk & mask
        n_correct_k = int(correct_topk.sum())
        return float(n_correct_k) / float(n_total)

    top_3_accuracy = compute_topk_accuracy(3)
    top_5_accuracy = compute_topk_accuracy(5)

    # Compute perplexity
    # Perplexity = exp(average cross-entropy loss)
    def compute_perplexity() -> float:
        """Compute perplexity at masked positions."""
        # Numerically stable softmax using log-sum-exp trick
        max_logits = predictions.max(axis=-1, keepdims=True)
        shifted = predictions - max_logits
        exp_logits = np.exp(shifted)
        sum_exp = exp_logits.sum(axis=-1, keepdims=True)
        log_probs = shifted - np.log(sum_exp)  # log softmax

        # Get log probabilities at label positions
        n_samples, seq_len, _ = predictions.shape

        # Create indices for gathering
        batch_indices = np.arange(n_samples)[:, np.newaxis]
        seq_indices = np.arange(seq_len)[np.newaxis, :]

        # Handle -100 labels by replacing with 0 temporarily (they'll be masked out)
        safe_labels = np.where(labels >= 0, labels, 0)

        # Gather log probs: log_probs[i, j, labels[i, j]]
        target_log_probs = log_probs[batch_indices, seq_indices, safe_labels]

        # Cross-entropy is negative log prob
        cross_entropy = -target_log_probs  # (n_samples, seq_len)

        # Average over masked positions only
        masked_ce = cross_entropy[mask]
        if len(masked_ce) == 0:
            return float("inf")

        avg_ce = float(masked_ce.mean())

        # Perplexity = exp(average cross-entropy)
        # Clip to avoid overflow
        if avg_ce > 100:
            return float("inf")

        return float(np.exp(avg_ce))

    perplexity = compute_perplexity()

    return {
        "accuracy": accuracy,
        "top_3_accuracy": top_3_accuracy,
        "top_5_accuracy": top_5_accuracy,
        "perplexity": perplexity,
    }
