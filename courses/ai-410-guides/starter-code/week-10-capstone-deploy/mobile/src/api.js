// Set in mobile/.env (copy from .env.example) as EXPO_PUBLIC_API_BASE —
// Expo inlines any EXPO_PUBLIC_* variable automatically, no extra config
// needed. Change the value, then reload the app (shake your phone →
// Reload, or press `r` in the terminal) to pick it up.
//
// TODO(week10): once your backend is deployed (see backend/render.yaml),
// update EXPO_PUBLIC_API_BASE in mobile/.env to the deployed URL, e.g.
// "https://ai410-api.onrender.com" — instead of your laptop's LAN IP.
export const API_BASE = process.env.EXPO_PUBLIC_API_BASE || "http://192.168.1.23:8000";

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
