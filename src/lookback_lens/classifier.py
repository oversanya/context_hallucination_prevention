"""
LookbackLensClassifier: logistic regression over lookback ratio features.

Wraps sklearn.linear_model.LogisticRegression with serialization helpers
and a threshold-aware evaluation method.

Reference: Chuang et al. (2024), Section 3.2.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class LookbackLensClassifier:
    """
    Binary classifier for hallucination detection from lookback ratio features.

    Label convention: 1 = hallucinated, 0 = faithful.

    Parameters
    ----------
    max_iter : int
        Maximum iterations for LogisticRegression solver.
    """

    def __init__(self, max_iter: int = 1000) -> None:
        self._clf = LogisticRegression(max_iter=max_iter, solver="lbfgs", C=1.0)
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LookbackLensClassifier":
        """
        Train the classifier on feature matrix X and binary labels y.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,) — binary int (0 or 1)
        """
        self._clf.fit(X, y)
        self._fitted = True
        logger.info(
            "Classifier fitted on %d samples, %d features.", X.shape[0], X.shape[1]
        )
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return hallucination probability for each sample.

        Returns
        -------
        np.ndarray, shape (n_samples,) — P(hallucinated | features)
        """
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba.")
        # Column 1 corresponds to positive class (hallucinated).
        return self._clf.predict_proba(X)[:, 1]

    # ── Evaluation ────────────────────────────────────────────────────────────

    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate hallucination detection on (X, y) using the optimal F1 threshold.

        Returns
        -------
        dict with keys: roc_auc, f1, precision, recall, optimal_threshold
        """
        if not self._fitted:
            raise RuntimeError("Classifier must be fitted before score.")

        proba = self.predict_proba(X)

        roc_auc = float(roc_auc_score(y, proba))

        # Find threshold that maximises F1 over the validation set.
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1, best_thresh = -1.0, 0.5
        for t in thresholds:
            y_pred = (proba >= t).astype(int)
            f1 = f1_score(y, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

        y_pred_opt = (proba >= best_thresh).astype(int)

        return {
            "roc_auc": roc_auc,
            "f1": float(best_f1),
            "precision": float(precision_score(y, y_pred_opt, zero_division=0)),
            "recall": float(recall_score(y, y_pred_opt, zero_division=0)),
            "optimal_threshold": best_thresh,
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> None:
        """Serialize the fitted classifier to disk via joblib."""
        if not self._fitted:
            raise RuntimeError("Cannot save an unfitted classifier.")
        joblib.dump(self, path)
        logger.info("Classifier saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LookbackLensClassifier":
        """Deserialize a previously saved LookbackLensClassifier."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected {cls}.")
        logger.info("Classifier loaded from %s", path)
        return obj
