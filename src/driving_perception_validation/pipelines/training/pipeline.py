"""
This is a boilerplate pipeline 'training'
generated using Kedro 1.7.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa

from .nodes import train_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
                func=train_model,
                inputs="params:training",
                outputs="training_result",
                name="train_model",
            ),
    ])
