"""
This is a boilerplate pipeline 'inference'
generated using Kedro 1.7.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa
from .nodes import run_inference


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
                func=run_inference,
                inputs=["images_manifest", "params:inference", "params:training"],
                outputs="predictions",
                name="run_inference",
        )
    ])
