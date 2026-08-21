# covidbench/models/xgboost_clf.py
from xgboost import XGBClassifier

from .. import config
from ..registry import register


@register(
    "xgboost",
    features=config.FEATURES_ALL,
    notes="Peer GBM sanity check",
    tracks=(config.COHORT_PAPER, config.COHORT_INCLUSIVE),
)
def xgboost_clf():
    return XGBClassifier(max_depth=4, n_estimators=300, eval_metric="logloss")