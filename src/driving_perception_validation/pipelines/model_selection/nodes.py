"""
This is a boilerplate pipeline 'model_selection'
generated using Kedro 1.7.0
"""

"""
Nodes for the model_selection pipeline: champion/challenger registry + test evaluation.

Flow
----
1. register_and_promote: register the newly trained model as a version of one
   registered model, give it the "challenger" alias, compare it with the current
   "champion" on the VALIDATION metric, and move the "champion" alias if it wins.
2. evaluate_champion_on_test: score the current champion once on the TEST split
   and log the numbers to the champion's MLflow run. The test split is never
   used to choose between models.
"""
import logging
from pathlib import Path
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def register_and_promote(training_result: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """Register the new model as a challenger and promote it if it beats the champion."""
    mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
    client = MlflowClient()

    name = params["registered_model_name"]
    metric = params["promotion_metric"]
    higher_is_better = params["higher_is_better"]
    min_delta = params["min_delta"]

    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name)

    run_id = training_result["run_id"]
    new_score = float(training_result["metrics"][metric])

    # Reuse the version if this run was already registered (safe to rerun the node).
    existing = [v for v in client.search_model_versions(f"name='{name}'") if v.run_id == run_id]
    if existing:
        mv = existing[0]
    else:
        mv = client.create_model_version(name=name, source=f"runs:/{run_id}/weights", run_id=run_id)
    for key in ("best_epoch", "val_loss", "val_map50"):
        client.set_model_version_tag(name, mv.version, key, str(round(training_result["metrics"][key], 6)))
    #client.set_model_version_tag(name, mv.version, metric, str(round(new_score, 6)))
    client.set_registered_model_alias(name, "challenger", mv.version)

    try:
        champion = client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        champion = None

    champion_score = None
    if champion is None:
        promote, reason = True, "no champion yet"
    elif champion.version == mv.version:
        promote, reason = False, "this version is already the champion"
    else:
        champion_score = client.get_run(champion.run_id).data.metrics.get(metric)
        if champion_score is None:
            raise ValueError(f"Champion run {champion.run_id} has no metric '{metric}'")
        delta = new_score - champion_score if higher_is_better else champion_score - new_score
        promote = delta > min_delta
        reason = f"{metric}: challenger {new_score:.4f} vs champion {champion_score:.4f}"

    if promote:
        client.set_registered_model_alias(name, "champion", mv.version)
        try:
            client.delete_registered_model_alias(name, "challenger")
        except MlflowException:
            pass
        decision = "promoted to champion"
    else:
        decision = "kept as challenger"
    client.set_model_version_tag(name, mv.version, "decision", decision)

    logger.info("Version %s %s (%s)", mv.version, decision, reason)
    return {
        "model_name": name,
        "version": str(mv.version),
        "run_id": run_id,
        "metric": metric,
        "challenger_score": new_score,
        "champion_score": champion_score,
        "decision": decision,
        "reason": reason,
    }


def evaluate_champion_on_test(promotion: dict[str, Any], params: dict[str, Any],
                              train_params: dict[str, Any]) -> dict[str, Any]:
    """Score the current champion on the TEST split and log it to its MLflow run."""
    mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
    client = MlflowClient()
    name = promotion["model_name"]

    champion = client.get_model_version_by_alias(name, "champion")
    run = client.get_run(champion.run_id)
    # Evaluate at the image size the champion was trained with.
    imgsz = int(run.data.params.get("imgsz", train_params["imgsz"]))

    weights = mlflow.artifacts.download_artifacts(run_id=champion.run_id, artifact_path="weights/best.pt")
    model = YOLO(weights)
    res = model.val(
        data=str(Path(train_params["data_yaml"]).resolve()),
        split="test",
        imgsz=imgsz,
        batch=train_params["batch"],
        device=train_params["device"],
        plots=False,
    )

    metrics = {
        "test_map50_95": float(res.box.map),
        "test_map50": float(res.box.map50),
        "test_precision": float(res.box.mp),
        "test_recall": float(res.box.mr),
    }
    per_class = {
        f"test_ap50_95_{res.names[i].replace(' ', '_')}": float(v)
        for i, v in enumerate(res.box.maps)
    }

    # Attach the test numbers to the champion's own run so everything is in one place.
    with mlflow.start_run(run_id=champion.run_id):
        mlflow.log_metrics({**metrics, **per_class})
    client.set_model_version_tag(name, champion.version, "test_map50_95", str(round(metrics["test_map50_95"], 6)))

    logger.info("Champion v%s on test: %s", champion.version, metrics)
    return {"model_name": name, "version": str(champion.version), "imgsz": imgsz,
            "metrics": metrics, "per_class_ap50_95": per_class}
