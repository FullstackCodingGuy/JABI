### **Integrating Gemma for Intent Recognition in a Personal Assistant**  

To use **Gemma** for **intent recognition** in your personal assistant, follow these steps:  

---

## **🔹 1. Approach to Intent Recognition**
### ✅ **Steps**:
1️⃣ **User Input** → The assistant receives a **text command**  
2️⃣ **Gemma Processes the Input** → The LLM **analyzes intent**  
3️⃣ **Intent Classification** → Matches the query to a **predefined action**  
4️⃣ **Execute Action** → Calls the function or API  
5️⃣ **Generate Response** → LLM confirms the action  

---

## **🔹 2. Running Gemma Locally for Intent Recognition**
Since **Gemma is an open-source model**, you can run it locally using **Ollama** or **Hugging Face Transformers**.

### **🛠️ Step 1: Install Ollama (For Local Inference)**
```sh
# Install Ollama (Mac/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Install Gemma model
ollama pull gemma:2b
```

### **🛠️ Step 2: Process User Queries with Intent Detection**
```typescript
import fetch from "node-fetch";

// Function to call Gemma LLM
async function callGemma(prompt: string): Promise<string> {
  const response = await fetch("http://localhost:11434/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: "gemma:2b", prompt })
  });

  const data = await response.json();
  return data.response.trim();
}

// Define Intent Handlers
const intentHandlers: { [key: string]: Function } = {
  "set_alarm": (time: string) => `Alarm set for ${time}.`,
  "send_email": (recipient: string, subject: string) =>
    `Email sent to ${recipient} with subject: ${subject}.`,
};

// Function to recognize intent
async function processIntent(userQuery: string): Promise<string> {
  const prompt = `
  Classify the intent of the following message into one of these intents: 
  - set_alarm
  - send_email
  - unknown

  Message: "${userQuery}"
  Response (JSON): 
  `;

  const response = await callGemma(prompt);
  console.log("LLM Response:", response);

  try {
    const result = JSON.parse(response);
    if (intentHandlers[result.intent]) {
      return intentHandlers[result.intent](...(result.parameters || []));
    }
  } catch (e) {
    console.error("Error parsing response:", e);
  }

  return "Sorry, I couldn't understand the request.";
}

// Example Usage
processIntent("Set an alarm for 6 AM").then(console.log);
processIntent("Send an email to john@example.com with subject Meeting").then(console.log);
```

✅ **Runs fully locally using Gemma 2B**  
✅ **Extracts intent & executes predefined functions**  

---

## **🔹 3. Fine-Tuning Gemma for Better Intent Recognition**
For **better accuracy**, train Gemma on **custom intent datasets**.

### **🛠️ Steps to Fine-Tune with Hugging Face**
1️⃣ **Prepare labeled dataset (text → intent mappings)**  
2️⃣ **Use Hugging Face `transformers` to fine-tune Gemma**  
3️⃣ **Deploy the fine-tuned model locally**  

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# Load Gemma model & tokenizer
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Example dataset (fine-tuning intent classification)
dataset = [
    {"input": "Set an alarm for 7 AM", "intent": "set_alarm"},
    {"input": "Remind me to call Alex", "intent": "set_reminder"},
    {"input": "Send an email to John with subject Urgent", "intent": "send_email"},
]

# Convert dataset into tokenized format
def tokenize_function(example):
    return tokenizer(example["input"], padding="max_length", truncation=True)

# Train model (if needed)
training_args = TrainingArguments(output_dir="./results", per_device_train_batch_size=4, num_train_epochs=3)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

trainer.train()
```

✅ **Fine-tunes Gemma for intent classification**  
✅ **Improves accuracy for specific use cases**  

---

## **🔹 4. Deploying in a Mobile App**
To integrate with a **mobile assistant (Flutter/React Native)**:
- **Backend**: Run Gemma on **FastAPI/Node.js**  
- **Frontend**: Send API requests for **intent recognition & action execution**  
- **Voice Integration**: Use **Google TTS / Whisper for voice input/output**  

---

