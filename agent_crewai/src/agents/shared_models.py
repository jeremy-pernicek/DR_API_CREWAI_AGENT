"""
Shared data models used across the multi-LLM system.
Contains common dataclasses to avoid circular imports.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass
class CodeSolution:
    """A code generation solution"""
    generator: str
    code: str
    explanation: str
    score: float = 0.0
    generation_time: float = 0.0
    

@dataclass
class GenerationMetrics:
    """Metrics for code generation process"""
    total_time: float = 0.0
    analysis_time: float = 0.0
    generation_time: float = 0.0
    scoring_time: float = 0.0
    synthesis_time: float = 0.0
    parallel_efficiency: float = 0.0  # speedup factor vs sequential


class SynthesisStrategy(Enum):
    """Strategy for code synthesis"""
    BEST_ONLY = "best_only"           # Use highest scoring solution as-is
    HYBRID_SYNTHESIS = "hybrid"       # Combine best components from multiple solutions
    ENHANCED_BEST = "enhanced"        # Enhance best solution with components from others
    RECOVERY_MODE = "recovery"        # All solutions poor, try to salvage best parts