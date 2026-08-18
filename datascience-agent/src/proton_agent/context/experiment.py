"""
Experiment Tracking: reproducible logging of dataset context, features, models, hyperparameters, and evaluation metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import uuid
from typing import Any, List, Optional

import pandas as pd


@dataclass
class ExperimentRun:
    """Encapsulates a single reproducible experiment run."""

    id: str
    timestamp: str
    dataset_name: str
    dataset_rows: int
    dataset_cols: int
    target_column: str
    feature_names: List[str]
    model_name: str
    problem_type: str
    metrics: dict[str, Any]
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "dataset_name": self.dataset_name,
            "shape": f"{self.dataset_rows} × {self.dataset_cols}",
            "target": self.target_column,
            "features_count": len(self.feature_names),
            "model": self.model_name,
            "problem_type": self.problem_type,
            "metrics": self.metrics,
        }


class ExperimentTracker:
    """Manages experiment logging and comparisons within the active session."""

    def __init__(self) -> None:
        self.runs: List[ExperimentRun] = []

    def log_run(
        self,
        dataset_name: str,
        dataset_rows: int,
        dataset_cols: int,
        target_column: str,
        feature_names: List[str],
        model_name: str,
        problem_type: str,
        metrics: dict[str, Any],
        parameters: Optional[dict[str, Any]] = None,
    ) -> ExperimentRun:
        run_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run = ExperimentRun(
            id=run_id,
            timestamp=timestamp,
            dataset_name=dataset_name,
            dataset_rows=dataset_rows,
            dataset_cols=dataset_cols,
            target_column=target_column,
            feature_names=feature_names,
            model_name=model_name,
            problem_type=problem_type,
            metrics=metrics,
            parameters=parameters or {},
        )
        self.runs.append(run)
        return run

    def list_runs(self) -> List[ExperimentRun]:
        return list(self.runs)

    def get_best_run(self, metric_name: str = "f1_score", higher_is_better: bool = True) -> Optional[ExperimentRun]:
        if not self.runs:
            return None
        valid_runs = [r for r in self.runs if metric_name in r.metrics]
        if not valid_runs:
            return self.runs[-1]
        return max(valid_runs, key=lambda r: r.metrics[metric_name]) if higher_is_better else min(valid_runs, key=lambda r: r.metrics[metric_name])

    def to_dataframe(self) -> pd.DataFrame:
        if not self.runs:
            return pd.DataFrame()
        records = []
        for r in self.runs:
            row = r.to_dict()
            metrics = row.pop("metrics")
            for k, v in metrics.items():
                row[f"metric_{k}"] = v
            records.append(row)
        return pd.DataFrame(records)

    def clear(self) -> None:
        self.runs.clear()
