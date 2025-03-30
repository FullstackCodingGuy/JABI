### **🔹 Functionalities Available from Gemma LLM (via HTTP API)**
When integrating **Gemma** via **HTTP API (Ollama or Hugging Face)**, you can utilize various functionalities to enhance your **AI assistant**.  

---

## **🔹 1. Core Functionalities**
### ✅ **1️⃣ Text Generation**
- **Generates human-like responses**  
- Example: **Answering general queries, summarization**  

🔹 **Example Request (HTTP POST)**:
```json
{
  "model": "gemma:2b",
  "prompt": "What are the benefits of AI in healthcare?",
  "max_tokens": 200
}
```
🔹 **Example Response**:
```json
{
  "response": "AI in healthcare helps with diagnostics, personalized treatment, and medical research."
}
```

---

### ✅ **2️⃣ Intent Recognition**
- **Detects user intent from text input**  
- Example: **Set alarm, schedule event, send email**  

🔹 **Example Request**:
```json
{
  "model": "gemma:2b",
  "prompt": "Classify intent for: 'Remind me to buy groceries at 5 PM'",
  "max_tokens": 100
}
```
🔹 **Example Response**:
```json
{
  "response": "{ 'intent': 'set_reminder', 'parameters': { 'time': '5 PM', 'task': 'buy groceries' }}"
}
```

---

### ✅ **3️⃣ Question Answering (QA)**
- **Extracts answers from context-based input**  
- Example: **"Who developed the Theory of Relativity?"**  

🔹 **Example Request**:
```json
{
  "model": "gemma:2b",
  "prompt": "Who developed the Theory of Relativity?",
  "max_tokens": 50
}
```
🔹 **Example Response**:
```json
{
  "response": "Albert Einstein developed the Theory of Relativity in the early 20th century."
}
```

---

### ✅ **4️⃣ Text Summarization**
- **Summarizes long texts into key points**  
- Example: **News articles, research papers, emails**  

🔹 **Example Request**:
```json
{
  "model": "gemma:2b",
  "prompt": "Summarize: 'Artificial intelligence is revolutionizing industries, improving efficiency, and automating tasks.'",
  "max_tokens": 50
}
```
🔹 **Example Response**:
```json
{
  "response": "AI improves industries by automating tasks and boosting efficiency."
}
```

---

### ✅ **5️⃣ Conversational Chatbot**
- **Acts as a chatbot for human-like interactions**  
- Example: **Casual chat, tech support, FAQs**  

🔹 **Example Request**:
```json
{
  "model": "gemma:2b",
  "prompt": "User: Hi! How’s the weather today? Assistant:",
  "max_tokens": 50
}
```
🔹 **Example Response**:
```json
{
  "response": "Hello! I can check the weather for you. Where are you located?"
}
```

---

### ✅ **6️⃣ Code Generation**
- **Generates code snippets in different programming languages**  
- Example: **JavaScript function, Python script, SQL query**  

🔹 **Example Request**:
```json
{
  "model": "gemma:2b",
  "prompt": "Write a Python function to reverse a string.",
  "max_tokens": 100
}
```
🔹 **Example Response**:
```json
{
  "response": "def reverse_string(s):\n    return s[::-1]\n\nprint(reverse_string('hello'))"
}
```

---

### ✅ **7️⃣ Text Classification**
- **Classifies input text into predefined categories**  
- Example: **Spam detection, sentiment analysis**  

🔹 **Example Request**:
```json
{
  "model": "gemma:2b",
  "prompt": "Classify: 'This is the best phone I’ve ever bought!' into categories: positive, negative, neutral.",
  "max_tokens": 50
}
```
🔹 **Example Response**:
```json
{
  "response": "{ 'category': 'positive' }"
}
```

---

### ✅ **8️⃣ Named Entity Recognition (NER)**
- **Extracts key entities (names, dates, locations, etc.)**  
- Example: **Identify entities in a user input**  

🔹 **Example Request**:
```json
{
  "model": "gemma:2b",
  "prompt": "Extract named entities from: 'Elon Musk founded Tesla in 2003 in the United States.'",
  "max_tokens": 100
}
```
🔹 **Example Response**:
```json
{
  "response": "{ 'person': 'Elon Musk', 'organization': 'Tesla', 'year': '2003', 'location': 'United States' }"
}
```

---

## **🔹 2. Advanced Capabilities**
### ✅ **9️⃣ Task Automation (Action Execution)**
- **Triggers predefined actions based on user intent**  
- Example: **Control smart home devices, schedule meetings**  

🔹 **Example Request**:
```json
{
  "model": "gemma:2b",
  "prompt": "User said: 'Turn off the living room lights' - Identify action.",
  "max_tokens": 50
}
```
🔹 **Example Response**:
```json
{
  "response": "{ 'intent': 'turn_off_light', 'parameters': { 'room': 'living room' } }"
}
```
✅ **Can be mapped to actual API calls** for smart home control.

---

### ✅ **🔟 Multilingual Translation**
- **Translates text between multiple languages**  
- Example: **English to Spanish translation**  

🔹 **Example Request**:
```json
{
  "model": "gemma:2b",
  "prompt": "Translate 'Hello, how are you?' into Spanish.",
  "max_tokens": 50
}
```
🔹 **Example Response**:
```json
{
  "response": "Hola, ¿cómo estás?"
}
```

---

## **🔹 3. How to Access Gemma LLM via HTTP**
### ✅ **Using Ollama API**
After installing Ollama:
```sh
curl -X POST http://localhost:11434/api/generate -d '{
  "model": "gemma:2b",
  "prompt": "Tell me a joke.",
  "max_tokens": 50
}'
```

### ✅ **Using TypeScript**
```typescript
import fetch from "node-fetch";

async function queryGemma(prompt: string) {
  const response = await fetch("http://localhost:11434/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "gemma:2b", prompt, max_tokens: 100 }),
  });

  const data = await response.json();
  console.log("Gemma Response:", data.response);
}

queryGemma("What is quantum computing?");
```

---

## **🚀 Conclusion**
🔹 **Gemma LLM can perform** intent recognition, text generation, question answering, classification, automation, and more.  
🔹 **Use cases include** AI chatbots, IT support, home automation, virtual assistants, and real-time content generation.  
🔹 **Easily integrate with HTTP API using Node.js, TypeScript, or Python**.
