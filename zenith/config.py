"""
Centralized configuration for all thresholds, weights, and window parameters.

Every tunable constant lives here. When migrating to ML, this module becomes
the parameter store that learned weights replace.
"""

from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainWeights:
    """Weights used to compose each domain score from raw inputs."""

    # Recovery = sleep_hour_score * w + recovery_raw * (1 - w)
    recovery_sleep: float = 0.6
    recovery_raw: float = 0.4

    # Emotional = mood * w + inverted_stress * (1 - w)
    emotional_mood: float = 0.7
    emotional_stress_inv: float = 0.3

    # Performance = completion * w + deep_work * (1 - w)
    performance_completion: float = 0.5
    performance_deep_work: float = 0.5

    # Energy/Focus = deep_work * w1 + mood * w2 + inverted_stress * w3
    energy_deep_work: float = 0.4
    energy_mood: float = 0.3
    energy_stress_inv: float = 0.3


@dataclass(frozen=True)
class GlobalWeights:
    """Weights for combining domain scores into global balance."""

    recovery: float = 0.25
    emotional: float = 0.25
    performance: float = 0.30
    energy_focus: float = 0.20

    def __post_init__(self):
        total = self.recovery + self.emotional + self.performance + self.energy_focus
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Global weights must sum to 1.0, got {total}")


# ---------------------------------------------------------------------------
# Scoring parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScoringParams:
    """Parameters for raw-to-score transformations."""

    # Sleep: optimal hours and penalty per hour deviation
    sleep_optimal_hours: float = 7.5
    sleep_penalty_per_hour: float = 2.0
    sleep_max_score: float = 10.0

    # Deep work: saturation cap in hours
    deep_work_cap_hours: float = 6.0

    # All scores are clamped to [0, scale_max]
    scale_max: float = 10.0


# ---------------------------------------------------------------------------
# Rolling / signal windows
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WindowParams:
    """Rolling window sizes for statistical computations."""

    short: int = 7
    long: int = 30
    min_periods: int = 1


# ---------------------------------------------------------------------------
# Volatility weights
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VolatilityWeights:
    """Weights for combining component standard deviations into volatility index."""

    mood: float = 0.4
    stress: float = 0.3
    deep_work: float = 0.3


# ---------------------------------------------------------------------------
# Trend classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrendThresholds:
    """Slope thresholds for labeling directional momentum."""

    strong_positive: float = 0.15
    weak_positive: float = 0.05
    weak_negative: float = -0.05
    strong_negative: float = -0.15
    min_data_points: int = 2


# ---------------------------------------------------------------------------
# Trend confidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrendConfidenceParams:
    """
    Weights and caps for the composite trend confidence score [0, 1].

    Confidence = w_r2 * R²  +  w_agreement * sign_agreement  -  w_volatility * vol_penalty

    - R²: goodness-of-fit of the short-window OLS (how linear is the trend)
    - sign_agreement: 1.0 if 7d and 30d slopes agree in sign, 0.0 otherwise
    - vol_penalty: min(volatility / vol_normalizer, 1.0) — higher vol → lower confidence
    """

    weight_r2: float = 0.50
    weight_agreement: float = 0.30
    weight_volatility: float = 0.20
    volatility_normalizer: float = 3.0   # volatility at which penalty saturates


# ---------------------------------------------------------------------------
# Detector thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BurnoutThresholds:
    """Thresholds for burnout flag detection."""

    low_recovery: float = 5.0
    high_performance: float = 7.0
    consecutive_days: int = 2      # current day + N-1 preceding days


@dataclass(frozen=True)
class CompensationThresholds:
    """Slope thresholds for detecting cross-domain compensation patterns."""

    rising_slope: float = 0.05
    falling_slope: float = -0.05


@dataclass(frozen=True)
class EarlyWarningThresholds:
    """Combined thresholds for early warning trigger."""

    deviation_floor: float = -0.5
    volatility_ceiling: float = 1.5


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegimeThresholds:
    """Boundary conditions for macro regime classification."""

    growth_slope: float = 0.05
    decline_slope: float = -0.05
    low_volatility: float = 1.2
    high_volatility: float = 1.5


# ---------------------------------------------------------------------------
# Compensation rules (declarative)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompensationRule:
    """A single compensation pattern: rising_domain masks falling_domain."""

    rising_domain: str
    falling_domain: str
    message: str


DEFAULT_COMPENSATION_RULES: tuple = (
    CompensationRule(
        rising_domain="performance",
        falling_domain="recovery",
        message="Performance compensating recovery decline",
    ),
    CompensationRule(
        rising_domain="performance",
        falling_domain="emotional",
        message="Performance compensating emotional decline",
    ),
    CompensationRule(
        rising_domain="energy_focus",
        falling_domain="recovery",
        message="Energy/focus compensating recovery decline",
    ),
    CompensationRule(
        rising_domain="performance",
        falling_domain="energy_focus",
        message="Performance compensating energy/focus decline",
    ),
)


# ---------------------------------------------------------------------------
# Top-level config aggregate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZenithConfig:
    """Complete engine configuration. Pass to pipeline to override defaults."""

    domain_weights: DomainWeights = field(default_factory=DomainWeights)
    global_weights: GlobalWeights = field(default_factory=GlobalWeights)
    scoring: ScoringParams = field(default_factory=ScoringParams)
    windows: WindowParams = field(default_factory=WindowParams)
    volatility: VolatilityWeights = field(default_factory=VolatilityWeights)
    trend: TrendThresholds = field(default_factory=TrendThresholds)
    trend_confidence: TrendConfidenceParams = field(default_factory=TrendConfidenceParams)
    burnout: BurnoutThresholds = field(default_factory=BurnoutThresholds)
    compensation: CompensationThresholds = field(default_factory=CompensationThresholds)
    early_warning: EarlyWarningThresholds = field(default_factory=EarlyWarningThresholds)
    regime: RegimeThresholds = field(default_factory=RegimeThresholds)
    compensation_rules: tuple = DEFAULT_COMPENSATION_RULES