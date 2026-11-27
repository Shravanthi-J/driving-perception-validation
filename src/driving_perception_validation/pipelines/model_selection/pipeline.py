"""
This is a boilerplate pipeline 'model_selection'
generated using Kedro 1.7.0
"""

from kedro.pipeline import Node, Pipeline, node  # noqa
from .nodes import register_and_promote, evaluate_champion_on_test

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
                func=register_and_promote,
                inputs=["training_result", "params:model_selection"],
                outputs="promotion_result",
                name="register_and_promote",
            ),
    ])
