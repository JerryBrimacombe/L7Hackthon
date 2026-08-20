"""Small neural-network benchmark using the shared eight-feature input."""
from __future__ import annotations

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config
from ..registry import register


@register(
    "neural_network",
    features=config.FEATURES_ALL,
    notes="Small multilayer perceptron",
)
def neural_network():
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=config.RANDOM_SEED,
        ),
    )