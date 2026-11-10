"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 1.7.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa
from .nodes import merge_label_files, summarize_conditions, items_to_tables, build_train_pool, build_test_pool, split_dataset, write_yolo_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
                func=merge_label_files,
                inputs="params:data_prep.train_labels_dir",
                outputs="bdd_labels_train",
                name="merge_train_labels",
            ),
            node(
                func=merge_label_files,
                inputs="params:data_prep.val_labels_dir",
                outputs="bdd_labels_val",
                name="merge_val_labels",
            ),
            node(
                func=summarize_conditions,
                inputs=["bdd_labels_train", "bdd_labels_val"],
                outputs="condition_counts",
                name="summarize_conditions",
            #2. records -> tables
            ),
            node(
                func=items_to_tables,
                inputs="bdd_labels_train",
                outputs=["train_images_raw", "train_boxes_raw"],
                name="train_to_tables"),
            node(
                func=items_to_tables,
                inputs="bdd_labels_val",
                outputs=["val_images_raw", "val_boxes_raw"],
                name="val_to_tables"),
            # 3. clean boxes, build pools
            node(
                func=build_train_pool, 
                inputs=["train_images_raw", "train_boxes_raw", "params:data_prep"],
                outputs=["train_pool_images", "train_pool_boxes"], 
                name="build_train_pool"),
            node(
                func=build_test_pool, 
                inputs=["val_images_raw", "val_boxes_raw", "params:data_prep"],
                outputs=["test_pool_images", "test_pool_boxes"], 
                name="build_test_pool"),
            # 4. stratified sampling and leakage-aware split
            node(
                func=split_dataset,
                inputs=["train_pool_images", "train_pool_boxes", "test_pool_images", "test_pool_boxes", "params:data_prep"],
                outputs=["images_manifest", "boxes_manifest"], 
                name="split_dataset",
                ),
            # 5. YOLO files on disk
            node(
                func=write_yolo_dataset, 
                inputs=["images_manifest", "boxes_manifest", "params:data_prep"],
                outputs="yolo_dataset_yaml", 
                name="write_yolo_dataset")
    ])
