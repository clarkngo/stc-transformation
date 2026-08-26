import Constants from "expo-constants";
import * as Device from "expo-device";
import * as Notifications from "expo-notifications";
import { useEffect, useRef, useState } from "react";
import {
  FlatList,
  KeyboardAvoidingView,
  Platform,
  SafeAreaView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";

import { registerDevice, sendChatMessage } from "./src/api";

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const deviceId = useRef(Constants.installationId || Device.osInternalBuildId || "dev-device").current;

  useEffect(() => {
    registerForPushNotificationsAsync();
  }, []);

  // Requesting permission and getting the token is already wired up —
  // this part is done for you.
  async function registerForPushNotificationsAsync() {
    if (!Device.isDevice) {
      console.warn("Push notifications require a physical device, not a simulator.");
      return;
    }

    const { status: existingStatus } = await Notifications.getPermissionsAsync();
    let finalStatus = existingStatus;
    if (existingStatus !== "granted") {
      const { status } = await Notifications.requestPermissionsAsync();
      finalStatus = status;
    }
    if (finalStatus !== "granted") {
      console.warn("Notification permission not granted.");
      return;
    }

    const { data: pushToken } = await Notifications.getExpoPushTokenAsync();
    console.log("Expo push token:", pushToken);

    // TODO(week8): send this token to your backend so it can push to
    // this specific device later. See src/api.js -> registerDevice().
    //
    // await registerDevice(deviceId, pushToken);
  }

  async function handleSend() {
    const text = input.trim();
    if (!text) return;

    setMessages((prev) => [...prev, { role: "user", text }]);
    setInput("");
    setSending(true);

    try {
      const reply = await sendChatMessage(text);
      setMessages((prev) => [...prev, { role: "assistant", text: reply }]);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", text: `⚠️ ${err.message}` }]);
    } finally {
      setSending(false);
    }
  }

  return (
    <SafeAreaView style={styles.container}>
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === "ios" ? "padding" : undefined}
      >
        <Text style={styles.header}>AI 410 — Chat</Text>

        <FlatList
          style={styles.messages}
          data={messages}
          keyExtractor={(_, i) => String(i)}
          renderItem={({ item }) => (
            <View style={[styles.bubble, item.role === "user" ? styles.userBubble : styles.assistantBubble]}>
              <Text style={styles.role}>{item.role}</Text>
              <Text>{item.text}</Text>
            </View>
          )}
        />

        <View style={styles.inputRow}>
          <TextInput
            style={styles.input}
            value={input}
            onChangeText={setInput}
            placeholder="Ask something…"
            editable={!sending}
          />
          <TouchableOpacity style={styles.sendButton} onPress={handleSend} disabled={sending}>
            <Text style={styles.sendButtonText}>Send</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#f7fafc" },
  header: { fontSize: 18, fontWeight: "700", color: "#1e8449", padding: 16, paddingBottom: 8 },
  messages: { flex: 1, paddingHorizontal: 12 },
  bubble: { padding: 10, borderRadius: 8, marginBottom: 8, maxWidth: "85%" },
  userBubble: { backgroundColor: "#dbeafe", alignSelf: "flex-end" },
  assistantBubble: { backgroundColor: "#fff", alignSelf: "flex-start", borderWidth: 1, borderColor: "#e2e8f0" },
  role: { fontSize: 10, fontWeight: "700", textTransform: "uppercase", color: "#64748b", marginBottom: 2 },
  inputRow: { flexDirection: "row", padding: 12, gap: 8 },
  input: { flex: 1, borderWidth: 1, borderColor: "#cbd5e1", borderRadius: 8, paddingHorizontal: 12, paddingVertical: 8, backgroundColor: "#fff" },
  sendButton: { backgroundColor: "#1e8449", borderRadius: 8, paddingHorizontal: 18, justifyContent: "center" },
  sendButtonText: { color: "#fff", fontWeight: "700" },
});
