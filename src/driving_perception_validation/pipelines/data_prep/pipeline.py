"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 1.7.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa
from .nodes import merge_label_files, summarize_conditions


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
    ),
    ])
