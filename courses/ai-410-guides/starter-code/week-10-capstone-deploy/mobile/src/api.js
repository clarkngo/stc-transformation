// TODO(week10): once your backend is deployed (see backend/render.yaml),
// point this at the deployed URL instead of your laptop's LAN IP, e.g.
// "https://ai410-api.onrender.com"
export const API_BASE = "http://192.168.1.23:8000";

export async function sendChatMessage(message) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  if (!res.ok) throw new Error(`Server returned ${res.status}`);
  const data = await res.json();
  return data.reply;
}

export async function registerDevice(deviceId, pushToken) {
  const res = await fetch(`${API_BASE}/register-device`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device_id: deviceId, push_token: pushToken }),
  });
  if (!res.ok) throw new Error(`Server returned ${res.status}`);
  return res.json();
}
