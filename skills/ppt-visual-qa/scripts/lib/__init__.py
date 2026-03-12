"""PPT Visual QA Library Modules."""

from .models import GateDef, ProfileMetrics, SlideFeatures, GateResult, SlideReport
from .constants import PROFILES_DEFAULT, MAIN_OUTER_AVAILABLE_PX
from .utils import strip_tags, count_structured_claims, infer_layout
from .feature_extractor import FeatureExtractor
from .profile_collector import ProfileCollector
from .gate_evaluator import GateEvaluator
from .runner import VisualQaRunner

__all__ = [
    "GateDef",
    "ProfileMetrics", 
    "SlideFeatures",
    "GateResult",
    "SlideReport",
    "PROFILES_DEFAULT",
    "MAIN_OUTER_AVAILABLE_PX",
    "strip_tags",
    "count_structured_claims",
    "infer_layout",
    "FeatureExtractor",
    "ProfileCollector",
    "GateEvaluator",
    "VisualQaRunner",
]