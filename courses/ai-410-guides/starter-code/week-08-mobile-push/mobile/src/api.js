// Points at your local backend. On a physical device, "localhost"
// means the PHONE, not your laptop — use your laptop's LAN IP
// instead (e.g. http://192.168.1.23:8000), or Expo's tunnel mode.
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

// TODO(week8): call this once you have a push token (see App.js) so
// the backend can send notifications to this specific device.
export async function registerDevice(deviceId, pushToken) {
  const res = await fetch(`${API_BASE}/register-device`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device_id: deviceId, push_token: pushToken }),
  });
  if (!res.ok) throw new Error(`Server returned ${res.status}`);
  return res.json();
}
