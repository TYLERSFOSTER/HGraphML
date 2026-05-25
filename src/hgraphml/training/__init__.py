"""Small training helpers for the first HGraphML proof."""

from hgraphml.training.objectives import make_teacher_node_targets
from hgraphml.training.train_step import TrainStepResult, train_step

__all__ = ["TrainStepResult", "make_teacher_node_targets", "train_step"]
