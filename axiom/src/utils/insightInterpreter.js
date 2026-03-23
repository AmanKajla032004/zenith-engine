const DRIVER_PROFILES = {
  performance: {
    title: "Performance Pressure",
    cause: "Performance demand is exceeding sustainable capacity.",
    effect: "This strains recovery systems and compresses emotional bandwidth.",
    consequence: {
      stable: "The system is absorbing the load, but margins are thin.",
      moderate_strain: "Sustained output pressure is eroding stability.",
      overloaded: "Performance demand has overwhelmed recovery and emotional capacity.",
      declining: "Prolonged performance strain has destabilized the system.",
    },
  },
  recovery: {
    title: "Recovery Deficit",
    cause: "Recovery capacity is below what the system requires.",
    effect: "Insufficient recovery amplifies fatigue and reduces cognitive resilience.",
    consequence: {
      stable: "Recovery is lagging but the system remains within safe bounds.",
      moderate_strain: "The recovery gap is widening, increasing fatigue risk.",
      overloaded: "Recovery failure is imminent under current load.",
      declining: "Chronic recovery deficit is driving system degradation.",
    },
  },
  energy: {
    title: "Energy Depletion",
    cause: "Energy availability is insufficient relative to workload demands.",
    effect: "Low energy reduces performance stability and accelerates fatigue accumulation.",
    consequence: {
      stable: "Energy reserves are low but the system is compensating.",
      moderate_strain: "Energy depletion is undermining sustained output.",
      overloaded: "Energy deficit has become the primary system bottleneck.",
      declining: "Severe energy depletion is cascading across all dimensions.",
    },
  },
  emotional: {
    title: "Emotional Strain",
    cause: "Emotional load is disproportionately high relative to other signals.",
    effect: "Emotional pressure suppresses recovery efficiency and distorts cognitive output.",
    consequence: {
      stable: "Emotional strain is present but contained.",
      moderate_strain: "Emotional load is beginning to interfere with system balance.",
      overloaded: "Emotional strain is destabilizing recovery and performance.",
      declining: "Emotional overload is the dominant force driving system decline.",
    },
  },
};

const DEFAULT_PROFILE = {
  title: "System Analysis",
  cause: "No dominant driver identified.",
  effect: "The system is operating without a clear imbalance signal.",
  consequence: {
    stable: "All dimensions are within expected ranges.",
    moderate_strain: "Minor strain detected across multiple dimensions.",
    overloaded: "System is under pressure from multiple sources.",
    declining: "System is declining without a single clear cause.",
  },
};

function buildSummary(driver, regime, pressure) {
  const driverLabel = DRIVER_PROFILES[driver]?.title || "mixed signals";
  const pressureLevel = pressure > 4 ? "critical" : pressure >= 2 ? "elevated" : "normal";

  const regimePhrases = {
    stable: `System is stable with ${pressureLevel} pressure from ${driverLabel.toLowerCase()}.`,
    moderate_strain: `System is under moderate strain driven by ${driverLabel.toLowerCase()}, with ${pressureLevel} pressure levels.`,
    overloaded: `System is overloaded due to ${driverLabel.toLowerCase()}, with ${pressureLevel} risk levels.`,
    declining: `System is in decline driven by ${driverLabel.toLowerCase()}. Pressure is ${pressureLevel}.`,
  };

  return regimePhrases[regime] || `System status: ${regime}. Driver: ${driverLabel.toLowerCase()}.`;
}

export function interpretInsights(dominant_driver, state, regime, pressure) {
  const profile = DRIVER_PROFILES[dominant_driver] || DEFAULT_PROFILE;
  const consequence = profile.consequence[regime] || profile.consequence.stable;

  return {
    title: profile.title,
    explanation: `${profile.cause} ${profile.effect}`,
    impact: consequence,
    summary: buildSummary(dominant_driver, regime, pressure),
  };
}