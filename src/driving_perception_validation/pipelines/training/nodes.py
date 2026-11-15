"""Nodes for the training pipeline: YOLO fine-tuning on Apple MPS with MLflow tracking.

Ultralytics-only version: this node opens the MLflow run, trains, logs the final
validation metrics and artifacts, and closes the run. No Kedro hook is needed.
"""
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def _best_epoch_losses(results_csv: Path) -> dict[str, float]:
    """Read validation losses at the epoch Ultralytics saved as best.pt.

    Ultralytics picks best.pt by fitness = 0.1 * mAP50 + 0.9 * mAP50-95.
    The three val losses are summed without weights into one rough number.
    """
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    fitness = 0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
    row = df.loc[fitness.idxmax()]
    box, cls, dfl = (float(row[f"val/{k}_loss"]) for k in ("box", "cls", "dfl"))
    return {
        "best_epoch": float(row["epoch"]),
        "val_box_loss": box,
        "val_cls_loss": cls,
        "val_dfl_loss": dfl,
        "val_loss": box + cls + dfl,
    }


def train_model(params: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune YOLO, track the run in MLflow, and return run info for later steps."""
    tracking_uri = params["mlflow_tracking_uri"]

    # Ultralytics' built-in MLflow callback reads these when training starts.
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_EXPERIMENT_NAME"] = params["experiment_name"]
    os.environ["MLFLOW_RUN"] = params["run_name"]
    # Keep the run open after training so we can log extra metrics to the same run.
    os.environ["MLFLOW_KEEP_RUN_ACTIVE"] = "true"
    # Let ops that MPS does not support fall back to CPU instead of crashing.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(params["experiment_name"])

    data_yaml = str(Path(params["data_yaml"]).resolve())
    project_dir = Path(params["project_dir"]).resolve()  # absolute, or Ultralytics prefixes runs/detect
    device = params["device"]

    with mlflow.start_run(run_name=params["run_name"]) as run:
        model = YOLO(params["model"])
        model.train(
            data=data_yaml,
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            workers=params["workers"],
            device=device,
            seed=params["seed"],
            fraction=params.get("fraction", 1.0),
            patience=params["patience"],
            project=str(project_dir),
            name=params["run_name"],
            exist_ok=True,
        )

        save_dir = Path(model.trainer.save_dir)
        best = save_dir / "weights" / "best.pt"

        # Evaluate the best weights on the VALIDATION split. This number is what we
        # compare between models. The test split is not touched here.
        best_model = YOLO(str(best))
        val = best_model.val(
            data=data_yaml, split="val", imgsz=params["imgsz"],
            batch=params["batch"], device=device, plots=False,
        )
        metrics = {
            "val_map50_95": float(val.box.map),
            "val_map50": float(val.box.map50),
            "val_precision": float(val.box.mp),
            "val_recall": float(val.box.mr),
        }
        metrics.update(_best_epoch_losses(save_dir / "results.csv"))
        mlflow.log_metrics(metrics)

        # Log the files we rely on explicitly, in case the built-in callback is disabled.
        mlflow.log_artifact(str(best), artifact_path="weights")
        mlflow.log_artifact(str(save_dir / "results.csv"))
        mlflow.set_tags({"data_yaml": data_yaml, "device": device})

        logger.info("Run %s: %s", run.info.run_id, metrics)
        return {
            "run_id": run.info.run_id,
            "run_name": params["run_name"],
            "best_weights": str(best),
            "tracking_uri": tracking_uri,
            "metrics": metrics,
        }