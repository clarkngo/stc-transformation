// AI 410 — frontend chat client.
// Points at the local backend. Change API_BASE if you deploy the
// backend elsewhere (see the Week 10 guide).
const API_BASE = "http://127.0.0.1:8000";

const form = document.getElementById("chat-form");
const input = document.getElementById("chat-input");
const submitBtn = document.getElementById("chat-submit");
const messages = document.getElementById("messages");

function addMessage(role, text) {
    const el = document.createElement("div");
    el.className = `msg ${role}`;
    el.innerHTML = `<span class="role">${role}</span>${escapeHtml(text)}`;
    messages.appendChild(el);
    messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
}

form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const message = input.value.trim();
    if (!message) return;

    addMessage("user", message);
    input.value = "";
    submitBtn.disabled = true;

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message }),
        });
        if (!res.ok) throw new Error(`Server returned ${res.status}`);
        const data = await res.json();
        addMessage("assistant", data.reply);
    } catch (err) {
        addMessage("assistant", `⚠️ Error talking to the backend: ${err.message}`);
    } finally {
        submitBtn.disabled = false;
        input.focus();
    }
});
