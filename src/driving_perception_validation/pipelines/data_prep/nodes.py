"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 1.7.0
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd

logger = logging.getLogger(__name__)

def merge_label_files(labels_dir: str) -> list[dict[str, Any]]:
    """Merge BDD100K per-image label JSONs into one list of image records."""

    files = sorted(Path(labels_dir).glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found in {labels_dir}")

    items, weather = [], Counter()
    for f in files:
        with open(f) as fh:
            d = json.load(fh)

        name = d.get("name") or f.stem
        if not name.endswith(".jpg"):
            name += ".jpg"

        objs = d["frames"][0].get("objects", []) if d.get("frames") else d.get("labels", [])
        labels = [
            {"category": o["category"], "box2d": o["box2d"], "attributes": o.get("attributes", {})}
            for o in objs
            if o.get("box2d") is not None  # skip lanes / drivable area
        ]
        attrs = d.get("attributes") or {}
        weather[attrs.get("weather", "missing")] += 1
        items.append({"name": name, "attributes": attrs, "labels": labels})

    logger.info("%s: %d images, weather=%s", labels_dir, len(items), dict(weather))
    return items


def summarize_conditions(train: list[dict], val: list[dict]) -> pd.DataFrame:
    """Count images per weather / scene / time of day for each split."""
    
    rows = []
    for split, items in [("train", train), ("val", val)]:
        n = len(items)
        for key in ["weather", "scene", "timeofday"]:
            counts = Counter(x["attributes"].get(key, "missing") for x in items)
            for value, c in counts.most_common():
                rows.append(
                    {"split": split, "attribute": key, "value": value,
                     "images": c, "pct": round(100 * c / n, 1)}
                )
    return pd.DataFrame(rows)

