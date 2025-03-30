> Response: I'm unable to schedule a Google Meet with David, as I do not have access to your calendar or scheduling capabilities.
Failed to parse JSON: I'm unable to schedule a Google Meet with David, as I do not have access to your calendar or scheduling capabilities.
 ⨯ Error: Invalid JSON format from LLM

### **🔹 Why Is This Happening?**  
The response is failing because:  
1. **LLM is responding conversationally** instead of returning structured JSON.  
2. **Gemma doesn’t have direct execution abilities** (e.g., booking meetings), so it assumes it must explain why.  
3. **System instructions are not strictly enforced**, leading to non-JSON responses.  

---

### **✅ Fix: Force Model to Return JSON**  
To ensure the response is **always valid JSON**, modify the **system instruction** and enforce **format constraints**.

#### **🔹 Step 1: Update System Instruction**  
Change the prompt to explicitly require **only JSON output**:  

```typescript
const SYSTEM_PROMPT = `
You are an intent recognition AI that extracts structured information from user requests.  
Always return a **valid JSON response**.  

## **Response Format (Always Follow This)**:
{
  "intent": "detected_intent",
  "parameters": { "key": "value" },
  "confidence": confidence_score
}

### **Rules:**
1️⃣ **DO NOT** explain anything.  
2️⃣ **DO NOT** include extra text.  
3️⃣ **DO NOT** say "I can't do that" – just extract intent.  
4️⃣ Always return a valid JSON object.  

## **Examples**
#### **User:** "Schedule a Google Meet with David tomorrow at 3 PM."
✅ **Response:**  
{
  "intent": "ScheduleMeeting",
  "parameters": {
    "date": "2025-04-01",
    "time": "3:00 PM",
    "participants": ["David"],
    "location": "Google Meet",
    "meetingType": "online"
  },
  "confidence": 0.97
}
`;

```

---

#### **🔹 Step 2: Call API with Strict Formatting**
Use this updated TypeScript function:

```typescript
async function callGemmaOllama(userInput: string) {
    const response = await fetch("http://localhost:11434/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            model: "gemma",
            messages: [
                { role: "system", content: SYSTEM_PROMPT },
                { role: "user", content: userInput }
            ],
            stream: false
        }),
    });

    const data = await response.json();

    try {
        return JSON.parse(data.response); // Ensure it's valid JSON
    } catch (error) {
        console.error("Failed to parse JSON:", data.response);
        throw new Error("Invalid JSON format from LLM");
    }
}

(async () => {
    const result = await callGemmaOllama("Schedule a Google Meet with David tomorrow at 3 PM.");
    console.log("Parsed Result:", result);
})();
```

---

### **✅ Expected Behavior**
This approach will **force the model to return structured JSON** instead of a conversational response.


---

### Fallback Mechanism

### **🔹 Adding Fallback Handling for Invalid JSON Responses**

To ensure robust handling of **invalid or unexpected responses**, you can implement a **fallback mechanism** that ensures your app doesn't break when the model returns something other than the expected JSON format.

Here's an updated TypeScript implementation that:

- **Validates the JSON response**
- **Returns a fallback message** if the response is invalid
- **Logs the error** for further debugging

---

### **✅ Step 1: Modify the TypeScript Function with Fallback Logic**

```typescript
async function callGemmaOllama(userInput: string) {
    try {
        const response = await fetch("http://localhost:11434/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model: "gemma",
                messages: [
                    { role: "system", content: SYSTEM_PROMPT },
                    { role: "user", content: userInput }
                ],
                stream: false
            }),
        });

        const data = await response.json();

        // Attempt to parse JSON response
        try {
            return JSON.parse(data.response);  // Ensure it's valid JSON
        } catch (jsonError) {
            console.error("Failed to parse JSON response:", data.response);
            // Fallback response when JSON is invalid
            return {
                intent: "Unknown",
                parameters: {},
                confidence: 0.0,
                message: "Unable to process the request. Please try again."
            };
        }
    } catch (networkError) {
        console.error("Network error or failed API request:", networkError);
        // Fallback response for network issues or errors during the API call
        return {
            intent: "Error",
            parameters: {},
            confidence: 0.0,
            message: "There was an issue communicating with the server. Please try again later."
        };
    }
}

(async () => {
    const result = await callGemmaOllama("Schedule a Google Meet with David tomorrow at 3 PM.");
    console.log("Parsed Result:", result);
})();
```

---

### **✅ Key Changes:**
1. **Fallback JSON Structure**:
   - If the response is invalid or non-JSON, the system will return a **default JSON** with `"intent": "Unknown"` and a helpful `message`.

2. **Error Handling for Network Failures**:
   - In case of network issues, it will return a fallback message like `"message": "There was an issue communicating with the server. Please try again later."`.

3. **Logging**:
   - Logs the actual response or error to the console, so you can debug if something goes wrong.

---

### **✅ Example Fallback Response**

If an invalid JSON or network error occurs, you would get a response like this:

```json
{
  "intent": "Unknown",
  "parameters": {},
  "confidence": 0.0,
  "message": "Unable to process the request. Please try again."
}
```

Or for network issues:

```json
{
  "intent": "Error",
  "parameters": {},
  "confidence": 0.0,
  "message": "There was an issue communicating with the server. Please try again later."
}
```

---

### **🔹 Benefits:**
✔ **Graceful Error Handling**: Ensures your app remains functional even when the model doesn’t respond as expected.  
✔ **User-Friendly Messages**: The user gets a meaningful fallback message rather than a crash or undefined behavior.  
✔ **Logging for Debugging**: You can trace the issues and improve the model integration further.  

✔ **Ensures proper JSON formatting**  
✔ **Prevents unnecessary explanations**  
✔ **Rejects invalid responses**  

Would you like to add **fallback handling** if the model still returns non-JSON? 🚀
