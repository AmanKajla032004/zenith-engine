const BASE_URL = "http://localhost:8000";

async function request(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;

  try {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => null);
      const message = errorBody?.detail || `Request failed: ${response.status}`;
      throw new Error(message);
    }

    return await response.json();
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error("Unable to reach the Zenith backend. Is the server running?");
    }
    throw error;
  }
}

export async function getHealth() {
  return request("/health");
}

export async function submitEntry(entry) {
  return request("/submit-entry", {
    method: "POST",
    body: JSON.stringify(entry),
  });
}

export async function getInsights() {
  return request("/insights");
}