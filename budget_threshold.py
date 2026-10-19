"""Thresholds for a fixed verification budget per batch, without training models."""
from decimal import Decimal, InvalidOperation, ROUND_FLOOR

import numpy as np
import pandas as pd


def _scores(probabilities):
    # Convert the input scores and validate their shape and range.
    """
    Validate and convert transaction scores to a numerical vector.

    Parameters
    ----------
        probabilities : array-like
            One-dimensional finite scores between zero and one.

    Returns
    -------
        scores : np.ndarray
            Validated floating-point scores.

    """
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Probabilidades devem ser um vetor finito entre 0 e 1.')
    return scores


def _money(value, name, positive=False):
    # Parse the monetary amount without binary floating-point rounding.
    """
    Validate a monetary amount using decimal arithmetic.

    Parameters
    ----------
        value : str, int, float, or Decimal
            Amount to validate, with at most two decimal places.

        name : str
            Field name used in validation messages.

        positive : bool
            Whether the amount must be strictly positive. Default is False.

    Returns
    -------
        number : Decimal
            Validated monetary amount.

    """
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f'{name} deve ser um valor monetário válido.') from None
    # Validate the sign, finiteness, and currency precision.
    if not number.is_finite() or number < 0 or (positive and number == 0):
        raise ValueError(f'{name} deve ser {"positivo" if positive else "não negativo"}.')
    if number != number.quantize(Decimal('0.01')):
        raise ValueError(f'{name} deve ter no máximo duas casas decimais.')
    return number


def threshold_for_budget(probabilities, verification_cost, budget):
    """
    Select the lowest inclusive score threshold that respects a batch budget.

    Parameters
    ----------
        probabilities : array-like
            Transaction scores between zero and one.

        verification_cost : str, int, float, or Decimal
            Positive cost per reviewed alert.

        budget : str, int, float, or Decimal
            Nonnegative review budget in the same currency and batch period.

    Returns
    -------
        result : dict
            Threshold, capacity, review count, spending, remaining budget, and selection mask.

    Notes
    -----
    Tied scores receive the same decision and can leave unused budget.
    An infinite threshold means no reviews. Every reviewed alert incurs a cost.

    """
    # Validate scores and monetary inputs before calculating review capacity.
    scores = _scores(probabilities)
    cost = _money(verification_cost, 'Custo por verificação', positive=True)
    available = _money(budget, 'Orçamento')
    capacity = int((available / cost).to_integral_value(rounding=ROUND_FLOOR))
    # Keep tied scores together when selecting the largest affordable group.
    if len(scores):
        unique, counts = np.unique(scores, return_counts=True)
        unique, counts = unique[::-1], counts[::-1]
        cumulative = counts.cumsum()
        eligible = np.flatnonzero(cumulative <= capacity)
        threshold = float(unique[eligible[-1]]) if len(eligible) else float('inf')
    else:
        threshold = float('inf')
    # Calculate the selected alerts and their actual review cost.
    selected = scores >= threshold
    reviews = int(selected.sum())
    spent = cost * reviews
    return {
        'threshold': threshold, 'capacity': capacity, 'reviews': reviews,
        'verification_cost': float(cost), 'budget': float(available),
        'spent': float(spent), 'remaining_budget': float(available-spent),
        'selected': selected,
    }


def cost_curve(y_true, probabilities, verification_cost):
    """
    Calculate observed precision, recall, and spending at every distinct score threshold.

    Parameters
    ----------
        y_true : array-like
            Binary fraud labels aligned with the transaction scores.

        probabilities : array-like
            Transaction scores between zero and one.

        verification_cost : str, int, float, or Decimal
            Positive cost per reviewed alert.

    Returns
    -------
        curve : pd.DataFrame
            Metrics at each threshold, including the no-review decision.

    """
    # Validate aligned labels, scores, and review cost.
    scores = _scores(probabilities)
    labels = np.asarray(y_true)
    if labels.ndim != 1 or len(labels) != len(scores) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Rótulos devem ser binários e alinhados às probabilidades.')
    cost = _money(verification_cost, 'Custo por verificação', positive=True)
    order = np.argsort(-scores, kind='stable')
    # Sort scores and evaluate only complete groups of tied scores.
    sorted_scores, labels = scores[order], labels[order].astype(int)
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(scores)-1] if len(scores) else np.array([], dtype=int)
    reviews = ends+1
    true_positive = labels.cumsum()[ends]
    total_fraud = int(labels.sum())
    result = pd.DataFrame({
        'threshold': sorted_scores[ends], 'reviews': reviews,
        'true_positive': true_positive, 'false_positive': reviews-true_positive,
        'false_negative': total_fraud-true_positive,
        'precision': true_positive/np.maximum(reviews, 1),
        'recall': true_positive/max(total_fraud, 1),
        'verification_spend': [float(cost*int(count)) for count in reviews],
    })
    # Build the metrics table and include the no-review baseline.
    empty = pd.DataFrame([{'threshold':float('inf'), 'reviews':0, 'true_positive':0,
        'false_positive':0, 'false_negative':total_fraud, 'precision':0., 'recall':0., 'verification_spend':0.}])
    return pd.concat([empty, result], ignore_index=True)
