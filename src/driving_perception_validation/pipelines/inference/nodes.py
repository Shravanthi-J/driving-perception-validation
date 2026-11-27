"""
This is a boilerplate pipeline 'inference'
generated using Kedro 1.7.0
"""

"""
Nodes for the inference pipeline: run the registry champion on images.

Produces
--------
- a table of per-image predictions (class, confidence, box in original pixels),
  which sliced evaluation and error analysis read later
- a few annotated sample images for specific conditions (night, rain), for
  looking at the predictions 
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

logger = logging.getLogger(__name__)

PRED_COLUMNS = ["name", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]


def _load_model(params: dict[str, Any], train_params: dict[str, Any]) -> tuple[YOLO, int, str]:
    """Download the weights of the model that holds the chosen alias and load them."""
    mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
    client = MlflowClient()
    mv = client.get_model_version_by_alias(params["registered_model_name"], params["alias"])
    run = client.get_run(mv.run_id)
    # Predict at the image size the model was trained with.
    imgsz = int(run.data.params.get("imgsz", train_params["imgsz"]))
    weights = mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="weights/best.pt")
    logger.info("Loaded %s@%s = version %s (run %s), imgsz=%d",
                params["registered_model_name"], params["alias"], mv.version, mv.run_id, imgsz)
    return YOLO(weights), imgsz, str(mv.version)


def _save_visuals(model: YOLO, images: pd.DataFrame, img_root: Path, imgsz: int,
                  params: dict[str, Any], device: str) -> None:
    """Save annotated images for a few conditions, e.g. night and rain."""
    out_dir = Path(params["visual_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for qi, query in enumerate(params["visual_queries"]):
        subset = images.query(query)
        if subset.empty:
            logger.warning("No images match visual query: %s", query)
            continue
        subset = subset.sample(n=min(params["n_per_query"], len(subset)), random_state=params["seed"])
        paths = [str(img_root / n) for n in subset.name]
        results = model.predict(source=paths, imgsz=imgsz, conf=params["vis_conf"],
                                device=device, verbose=False)
        for r in results:
            cv2.imwrite(str(out_dir / f"q{qi}_{Path(r.path).name}"), r.plot())  # r.plot() returns BGR
    logger.info("Annotated samples saved to %s", out_dir)


def run_inference(images_manifest: pd.DataFrame, params: dict[str, Any],
                  train_params: dict[str, Any]) -> pd.DataFrame:
    """Run the aliased model on one split and return every detection as a table row."""
    device = train_params["device"]
    model, imgsz, version = _load_model(params, train_params)

    images = images_manifest[images_manifest["split"] == params["split"]]
    if params.get("max_images"):
        images = images.sample(n=min(params["max_images"], len(images)), random_state=params["seed"])
    img_root = Path(params["yolo_dir"]) / "images" / params["split"]
    paths = [str(img_root / n) for n in images.name]
    logger.info("Predicting on %d %s images", len(paths), params["split"])

    rows = []
    chunk = params["chunk_size"]
    for i in range(0, len(paths), chunk):
        results = model.predict(source=paths[i:i + chunk], imgsz=imgsz, conf=params["conf"],
                                device=device, verbose=False)
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()   # pixels of the ORIGINAL image
            conf = r.boxes.conf.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            name = Path(r.path).name
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                rows.append((name, int(k), r.names[int(k)], float(c),
                             float(x1), float(y1), float(x2), float(y2)))
        logger.info("  %d / %d images done", min(i + chunk, len(paths)), len(paths))

    preds = pd.DataFrame(rows, columns=PRED_COLUMNS)
    preds["split"] = params["split"]
    preds["model_version"] = version
    logger.info("%d detections on %d images", len(preds), len(paths))

    _save_visuals(model, images, img_root, imgsz, params, device)
    return preds