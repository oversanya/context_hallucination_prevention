"""
LookbackLensClassifier: logistic regression over lookback ratio features.

Wraps sklearn.linear_model.LogisticRegression(CV) with serialization helpers
and a threshold-aware evaluation method.

When ``use_cv=True`` (default), the classifier uses ``LogisticRegressionCV``
with an internal stratified k-fold to select ``C`` and supports L1 sparsity.
This is the recommended setting for high-dimensional features (e.g., Qwen-7B
attention with 28 layers x 28 heads = 784 features and only ~160 train rows).

Reference: Chuang et al. (2024), Section 3.2.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
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
        Maximum iterations for the solver.
    use_cv : bool
        If True, use ``LogisticRegressionCV`` with a stratified k-fold to pick
        ``C`` from ``Cs`` automatically.  Recommended for high-dim features.
    Cs : list of float, optional
        Inverse regularisation strengths to search over.  Default log-grid.
    penalty : {"l1", "l2"}
        Regularisation norm.  L1 yields sparse solutions, useful when many
        attention heads carry no signal.
    cv : int
        Number of folds for cross-validation when ``use_cv`` is True.
    """

    def __init__(
        self,
        max_iter: int = 5_000,
        use_cv: bool = True,
        Cs: Optional[list] = None,
        penalty: str = "l1",
        cv: int = 5,
    ) -> None:
        self.use_cv     = use_cv
        self.penalty    = penalty
        self.cv         = cv
        self.Cs         = Cs if Cs is not None else [1e-3, 1e-2, 0.1, 1.0, 10.0]

        if use_cv:
            # liblinear supports L1; saga is more flexible but slower.
            solver = "liblinear" if penalty == "l1" else "lbfgs"
            self._clf = LogisticRegressionCV(
                Cs=self.Cs,
                cv=cv,
                penalty=penalty,
                solver=solver,
                max_iter=max_iter,
                scoring="roc_auc",
                refit=True,
                n_jobs=1,
            )
        else:
            solver = "liblinear" if penalty == "l1" else "lbfgs"
            self._clf = LogisticRegression(
                max_iter=max_iter, solver=solver, C=1.0, penalty=penalty
            )
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
        if self.use_cv:
            chosen_C = float(self._clf.C_[0]) if hasattr(self._clf, "C_") else None
            n_nonzero = int(np.sum(np.abs(self._clf.coef_) > 1e-8))
            logger.info(
                "LogisticRegressionCV — chosen C=%s, nonzero coefficients=%d/%d",
                chosen_C, n_nonzero, X.shape[1],
            )
        return self

    @property
    def chosen_C(self) -> Optional[float]:
        """Selected regularisation strength (only meaningful when use_cv=True and fitted)."""
        if not self._fitted or not hasattr(self._clf, "C_"):
            return None
        return float(self._clf.C_[0])

    @property
    def n_nonzero_coef(self) -> Optional[int]:
        """Number of non-zero coefficients (sparsity proxy under L1)."""
        if not self._fitted or not hasattr(self._clf, "coef_"):
            return None
        return int(np.sum(np.abs(self._clf.coef_) > 1e-8))

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
