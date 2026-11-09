"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 1.7.0
"""

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, List
import pandas as pd
import numpy as np

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

def items_to_tables(items: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn image json records into an images table and a boxes table."""
    
    imgs, boxes = [], []
    for item in items:
        name = item["name"]
        attrs = item.get("attributes") or {}
        imgs.append(
            dict(
                name=name,
                video_id=Path(name).stem.split("-")[0],  # prefix = source video
                weather=attrs.get("weather", "undefined"),
                scene=attrs.get("scene", "undefined"),
                timeofday=attrs.get("timeofday", "undefined"),
            )
        )
        for label in item.get("labels") or []:
            box_coords = label.get("box2d")
            if box_coords is None:
                continue
            label_attrs = label.get("attributes") or {}
            boxes.append(
                dict(
                    name=name, category=label["category"],
                    x1=box_coords["x1"], y1=box_coords["y1"], x2=box_coords["x2"], y2=box_coords["y2"],
                    occluded=bool(label_attrs.get("occluded", False)),
                    truncated=bool(label_attrs.get("truncated", False)),
                    traffic_light_color=label_attrs.get("trafficLightColor", "none"),
                )
            )
    #box_cols = ["name", "category", "x1", "y1", "x2", "y2", "occluded", "truncated"]
    return pd.DataFrame(imgs), pd.DataFrame(boxes), #columns=box_cols)

def _build_pool(images:pd.DataFrame, boxes:pd.DataFrame, images_dir: Path, classes:List, min_box_px:float,IMG_W:int, IMG_H:int):
    """
    Clips the bounding boxes to the image dimensions and 
    filters out boxes that are too small less than min_box_px (height and width not less than min_box_px pixels).
    Take only those images that have boxes
    Check if the images exist in the images directory. If not, clean those images from the images and box dataframes.
    Add a stratum column to the images dataframe.
    Return the images and boxes dataframes.
    """

    box_df = boxes[boxes.category.isin(classes)].copy()
    box_df[["x1", "x2"]] = box_df[["x1", "x2"]].clip(0, IMG_W)
    box_df[["y1", "y2"]] = box_df[["y1", "y2"]].clip(0, IMG_H)
    box_df["w"] = box_df.x2 - box_df.x1
    box_df["h"] = box_df.y2 - box_df.y1
    box_df["area"] = box_df.w * box_df.h
    box_df = box_df[(box_df.w >= min_box_px) & (box_df.h >= min_box_px)].reset_index(drop=True)
 
    imgs_df = images[images.name.isin(box_df.name)].copy()
    exists = imgs_df.name.map(lambda n: (images_dir / n).exists())
    n_missing = int((~exists).sum())
    if n_missing:
        logger.warning("%d labelled images not found in %s; skipping them.", n_missing, images_dir)
    imgs_df = imgs_df[exists].reset_index(drop=True)
    if imgs_df.empty:
        raise FileNotFoundError(
            f"No labelled images found in {images_dir}. Check data_prep.images_root in parameters."
        )
    box_df = box_df[box_df.name.isin(imgs_df.name)].reset_index(drop=True)
    imgs_df["stratum"] = imgs_df.timeofday + "|" + imgs_df.weather
    logger.info("Pool from %s: %d images, %d boxes", images_dir, len(imgs_df), len(box_df))
    return imgs_df, box_df

def build_train_pool(images:pd.DataFrame, boxes:pd.DataFrame, params: dict):
    """Pool for train+val sampling (official TRAIN images)."""
    return _build_pool(images, boxes, Path(params["images_dir"]) / "train",
                       params["classes"], params["min_box_px"],params["IMG_W"], params["IMG_H"])


def build_test_pool(images:pd.DataFrame, boxes:pd.DataFrame, params: dict):
    """Pool for the held-out test set (official VAL images)."""
    return _build_pool(images, boxes, Path(params["images_dir"]) / "val",
                       params["classes"], params["min_box_px"], params["IMG_W"], params["IMG_H"])

def _stratified_sample(df:pd.DataFrame, no_of_rows:int, seed:int):
    """Sample ~n rows keeping the (timeofday|weather) mix; rare strata keep >=1."""

    rng = np.random.default_rng(seed)
    no_of_rows = min(no_of_rows, len(df))
    frac = no_of_rows / len(df)
    sampled_data = []
    for _, group_data in df.groupby("stratum"):
        k = min(len(group_data), max(1, int(round(len(group_data) * frac)))) #when rounded grouped datapoints can become 0 so at max 1 need to be taken
        sampled_data.append(group_data.sample(n=k, random_state=int(rng.integers(1_000_000_000))))
    out = pd.concat(sampled_data)
    if len(out) > no_of_rows:
        output_df = out.sample(n=no_of_rows, random_state=seed)
    return output_df.reset_index(drop=True)

def _group_split(df:pd.DataFrame, val_frac:float, seed:int):
    """
    Split by video id within each stratum so no video spans train and val.
    val_frac = 0.2. Two videos go to val, eight go to train, and all frames of each video follow it
    Returns data frames : One for validation and one for train
    """
    rng = np.random.default_rng(seed)
    val_videos = set()
    for _, g in df.groupby("stratum"):
        vids = g.video_id.unique()
        rng.shuffle(vids)
        k = int(round(len(vids) * val_frac))
        val_videos.update(vids[:k])
    is_val = df.video_id.isin(val_videos)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)

def _check_no_leakage(splits: dict):
    names = list(splits)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = set(splits[names[i]].video_id) & set(splits[names[j]].video_id)
            assert not overlap, f"Video leakage between {names[i]} and {names[j]}: {len(overlap)} videos"
    logger.info("Leakage check passed: no video id appears in more than one split.")
 
 
def split_dataset(train_imgs:pd.DataFrame, train_boxes:pd.DataFrame, test_imgs:pd.DataFrame, test_boxes:pd.DataFrame, params: dict):
    """Build train / val / test manifests (images and boxes, with a `split` column)."""
    n_train, n_val, n_test, seed = (params[k] for k in ("n_train", "n_val", "n_test", "seed"))
 
    sampled = _stratified_sample(train_imgs, n_train + n_val, seed)
    train, val = _group_split(sampled, n_val / (n_train + n_val), seed)
    test = _stratified_sample(test_imgs, n_test, seed + 1)
 
    # (images, boxes pool, official image folder the files live in)
    splits = {
        "train": (train, train_boxes, "train"),
        "val": (val, train_boxes, "train"),
        "test": (test, test_boxes, "val"),
    }
    _check_no_leakage({k: v[0] for k, v in splits.items()})
 
    img_parts, box_parts = [], []
    for split, (imgs_df, pool_boxes, src_dir) in splits.items():
        box_df = pool_boxes[pool_boxes.name.isin(imgs_df.name)]
        counts = box_df.groupby("name").size()
        img_parts.append(imgs_df.assign(
            split=split, source_dir=src_dir,
            n_boxes=imgs_df.name.map(counts).fillna(0).astype(int)))
        box_parts.append(box_df.assign(split=split))
        logger.info("%s: %d images, %d boxes", split, len(imgs_df), len(box_df))
 
    return pd.concat(img_parts, ignore_index=True), pd.concat(box_parts, ignore_index=True)