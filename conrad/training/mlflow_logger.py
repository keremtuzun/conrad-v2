"""Optional, local-file MLflow logging. It can never fail or block a run.

The tracking store is a directory (default ``artifacts/mlruns``); no server and no network. When
disabled, MLflow is not even imported.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class MlflowLogger:
    def __init__(
        self, *, enabled: bool, tracking_dir: str | Path = "artifacts/mlruns", experiment: str = "conrad-v2"
    ) -> None:
        self.enabled = enabled
        self.tracking_dir = Path(tracking_dir)
        self.experiment = experiment
        self.error: str | None = None
        self._mlflow: Any = None
        self._active = False

    def _fail(self, exc: Exception) -> None:
        self.error = f"{type(exc).__name__}: {exc}"
        self.enabled = False
        self._active = False

    def start(self, run_name: str, params: Mapping[str, Any] | None = None) -> bool:
        if not self.enabled:
            return False
        try:
            self._mlflow = importlib.import_module("mlflow")
            self.tracking_dir.mkdir(parents=True, exist_ok=True)
            self._mlflow.set_tracking_uri(self.tracking_dir.resolve().as_uri())
            self._mlflow.set_experiment(self.experiment)
            self._mlflow.start_run(run_name=run_name)
            self._active = True
            if params:
                self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        except Exception as exc:  # logging is best effort by contract; the reason is kept in .error
            self._fail(exc)
        return self._active

    def log_metrics(self, record: Mapping[str, Any]) -> None:
        """Usable directly as a Trainer metric sink."""
        if not self._active:
            return
        try:
            step = int(record.get("step", 0))
            numeric = {
                k: float(v)
                for k, v in record.items()
                if isinstance(v, int | float) and not isinstance(v, bool)
            }
            self._mlflow.log_metrics(numeric, step=step)
        except Exception as exc:
            self._fail(exc)

    def end(self, status: str = "FINISHED") -> None:
        if not self._active:
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception as exc:
            self._fail(exc)
        self._active = False
