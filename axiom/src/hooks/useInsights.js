import { useState, useEffect, useCallback } from "react";
import { getInsights } from "../api/zenithService";
import { interpretInsights } from "../utils/insightInterpreter";

const MAX_HISTORY = 20;

export function useInsights() {
  const [insights, setInsights] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [insightsHistory, setInsightsHistory] = useState([]);

  const fetchInsights = useCallback(async (isInitial) => {
    if (isInitial) setLoading(true);
    setError(null);

    try {
      const data = await getInsights();
      data.state_vector = [
        data.state.performance,
        data.state.recovery,
        data.state.energy,
        data.state.emotional,
      ];
      data.interpretation = interpretInsights(
        data.dominant_driver,
        data.state,
        data.regime,
        data.behavioral_pressure_index
      );
      setInsights(data);

      setInsightsHistory((prev) => {
        const entry = {
          date: new Date().toLocaleTimeString(),
          balance_score: data.balance_score,
          performance: data.state.performance,
          recovery: data.state.recovery,
          energy: data.state.energy,
          emotional: data.state.emotional,
        };
        const updated = [...prev, entry];
        return updated.slice(-MAX_HISTORY);
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchInsights(true);
  }, [fetchInsights]);

  const refreshInsights = useCallback(() => fetchInsights(false), [fetchInsights]);

  return { insights, loading, error, refreshInsights, insightsHistory };
}