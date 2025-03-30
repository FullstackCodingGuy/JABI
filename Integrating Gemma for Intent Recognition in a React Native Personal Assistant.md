### **Integrating Gemma for Intent Recognition in a React Native Personal Assistant**  

This guide will help you **integrate Gemma** into a **React Native** mobile assistant that can recognize user intents and perform actions.  

---

## **🔹 1. Architecture Overview**
### ✅ **Flow of the Assistant**  
1️⃣ **User speaks or types a request**  
2️⃣ **React Native sends the request to a backend (Node.js/FastAPI)**  
3️⃣ **Gemma processes the text & detects intent**  
4️⃣ **Backend executes the mapped action** (e.g., set an alarm, send a message)  
5️⃣ **Response is sent back to the mobile app**  

---

## **🔹 2. Backend: Node.js + Gemma for Intent Recognition**
First, we **set up a backend** using **Node.js + Express** to handle user queries.

### **🛠️ Step 1: Install Dependencies**
```sh
# Install dependencies
npm init -y
npm install express cors body-parser node-fetch
```

### **🛠️ Step 2: Create `server.js` for Handling Requests**
```javascript
import express from "express";
import cors from "cors";
import fetch from "node-fetch";

const app = express();
app.use(cors());
app.use(express.json());

const GEMMA_API_URL = "http://localhost:11434/api/generate"; // Ollama Server

// Function to call Gemma for intent recognition
async function callGemma(prompt) {
  const response = await fetch(GEMMA_API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "gemma:2b", prompt })
  });

  const data = await response.json();
  return data.response.trim();
}

// Define action handlers
const intentHandlers = {
  "set_alarm": (params) => `Alarm set for ${params.time}.`,
  "send_message": (params) => `Message sent to ${params.recipient}: "${params.message}".`
};

// API Endpoint for Processing User Requests
app.post("/process-intent", async (req, res) => {
  const { userQuery } = req.body;
  
  // Prompt for intent detection
  const prompt = `
  Identify the intent of the following message and return JSON:
  - set_alarm
  - send_message
  - unknown

  Message: "${userQuery}"
  Response (JSON): 
  `;

  const response = await callGemma(prompt);
  console.log("Gemma Response:", response);

  try {
    const result = JSON.parse(response);
    if (intentHandlers[result.intent]) {
      return res.json({ response: intentHandlers[result.intent](result.parameters) });
    }
  } catch (error) {
    console.error("Error parsing response:", error);
  }

  res.json({ response: "Sorry, I couldn't understand your request." });
});

// Start Server
app.listen(3000, () => console.log("Server running on port 3000"));
```

✅ **Handles user input & classifies intent**  
✅ **Calls Gemma for LLM-based understanding**  
✅ **Executes matched actions & returns responses**  

---

## **🔹 3. React Native App to Connect with Backend**
Now, we create a **React Native front-end** to send user queries and display responses.

### **🛠️ Step 1: Install React Native & Required Packages**
```sh
npx react-native init PersonalAssistant
cd PersonalAssistant
npm install axios react-native-voice
```

### **🛠️ Step 2: Create `IntentAssistant.tsx` Component**
```tsx
import React, { useState } from "react";
import { View, Text, TextInput, Button, ScrollView } from "react-native";
import axios from "axios";
import Voice from "@react-native-voice/voice"; // Voice-to-Text

export default function IntentAssistant() {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");
  const [isListening, setIsListening] = useState(false);

  // Handle text-based queries
  const handleSend = async () => {
    if (!input.trim()) return;
    try {
      const res = await axios.post("http://localhost:3000/process-intent", { userQuery: input });
      setResponse(res.data.response);
    } catch (error) {
      setResponse("Error processing request.");
    }
  };

  // Handle voice input
  const startListening = async () => {
    setIsListening(true);
    Voice.start("en-US");
    Voice.onSpeechResults = (event) => {
      setInput(event.value[0]); // Set transcribed text
      setIsListening(false);
    };
  };

  return (
    <View style={{ flex: 1, padding: 20 }}>
      <Text style={{ fontSize: 22, fontWeight: "bold" }}>AI Personal Assistant</Text>
      
      <TextInput
        value={input}
        onChangeText={setInput}
        placeholder="Type or speak your request..."
        style={{ borderWidth: 1, padding: 10, marginVertical: 10 }}
      />

      <Button title="Send" onPress={handleSend} />
      <Button title={isListening ? "Listening..." : "Speak"} onPress={startListening} />

      <ScrollView style={{ marginTop: 20 }}>
        <Text style={{ fontSize: 18 }}>{response}</Text>
      </ScrollView>
    </View>
  );
}
```

✅ **Sends queries to the backend**  
✅ **Displays responses from the assistant**  
✅ **Supports both text and voice input**  

---

## **🔹 4. Running the Mobile Assistant**
### **1️⃣ Start the Backend**
```sh
node server.js
```

### **2️⃣ Run React Native App**
For Android:
```sh
npx react-native run-android
```
For iOS:
```sh
npx react-native run-ios
```

✅ **Your AI-powered mobile assistant is ready!**  

---

## **🔹 5. Enhancing the Assistant**
🔹 **Action Execution**: Add API calls for home automation, reminders, etc.  
🔹 **Offline Mode**: Use **Gemma with Ollama** on the device.  
🔹 **Voice Output**: Integrate **Text-to-Speech (TTS)**.  
🔹 **Integration with Chat UI**: Use **React Native Gifted Chat**.  
