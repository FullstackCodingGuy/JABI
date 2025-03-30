### **Running Gemma 2B on Mobile Using ONNX and Interacting with React Native**  

To run **Gemma 2B** on **mobile (Android/iOS)** using **ONNX** and interact with it in **React Native**, follow this guide.

---

## **🔹 Step 1: Convert Gemma 2B to ONNX**  
Gemma 2B is a **large model** and must be converted to an **ONNX format** for mobile compatibility.  

### **1️⃣ Install Necessary Dependencies**
Use **transformers** and **onnxruntime** to convert the model:
```sh
pip install transformers onnxruntime optimum
```

### **2️⃣ Convert Gemma 2B to ONNX**
Run the script to convert the Hugging Face **Gemma model**:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

model_name = "google/gemma-2b"

# Load the Gemma model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Export to ONNX
onnx_model_path = "gemma-2b.onnx"
ORTModelForCausalLM.from_pretrained(model, save_directory=onnx_model_path, export=True)

print(f"Gemma 2B converted to ONNX at {onnx_model_path}")
```
This converts **Gemma 2B** into an **ONNX model (`gemma-2b.onnx`)**.

---

## **🔹 Step 2: Set Up React Native with ONNX Runtime**
### **1️⃣ Create a React Native Project**
```sh
npx react-native init GemmaAssistant
cd GemmaAssistant
```

### **2️⃣ Install ONNX Runtime for React Native**
```sh
npm install onnxruntime-react-native
```

---

## **🔹 Step 3: Add ONNX Model to React Native**
Place the **`gemma-2b.onnx`** file inside your project:
```
/assets/gemma-2b.onnx
```

For **Android**, add it under:
```
android/app/src/main/assets/gemma-2b.onnx
```

For **iOS**, update `Info.plist`:
```xml
<key>UIFileSharingEnabled</key>
<true/>
<key>LSSupportsOpeningDocumentsInPlace</key>
<true/>
```

---

## **🔹 Step 4: Load and Run Gemma 2B ONNX Model in React Native**
### **Create `gemma-runner.js`**
```javascript
import { OrtSession, Tensor } from 'onnxruntime-react-native';
import { Platform, NativeModules } from 'react-native';

// Load ONNX model
export async function loadGemma() {
    const modelPath = Platform.OS === 'ios'
        ? `${NativeModules.NativeUtils.getDocumentDirectory()}/gemma-2b.onnx`
        : 'file:///android_asset/gemma-2b.onnx';

    const session = await OrtSession.create(modelPath);
    return session;
}

// Run inference using Gemma 2B ONNX
export async function runGemma(session, inputText) {
    try {
        const inputTensor = new Tensor('int32', new Int32Array(inputText.split("").map(c => c.charCodeAt(0))), [1, inputText.length]);

        const feeds = { input: inputTensor };
        const results = await session.run(feeds);

        const outputText = results.output.data.map(charCode => String.fromCharCode(charCode)).join("");
        console.log("Gemma Output:", outputText);
        return outputText;
    } catch (error) {
        console.error("Gemma Inference Error:", error);
        return null;
    }
}
```

---

## **🔹 Step 5: Integrate Gemma in React Native UI**
### **Modify `App.js`**
```javascript
import React, { useEffect, useState } from 'react';
import { View, Text, TextInput, Button, StyleSheet } from 'react-native';
import { loadGemma, runGemma } from './gemma-runner';

export default function App() {
    const [gemmaModel, setGemmaModel] = useState(null);
    const [input, setInput] = useState('');
    const [output, setOutput] = useState('');

    useEffect(() => {
        async function init() {
            const session = await loadGemma();
            setGemmaModel(session);
        }
        init();
    }, []);

    const handleRunGemma = async () => {
        if (!gemmaModel) return;
        const response = await runGemma(gemmaModel, input);
        setOutput(response);
    };

    return (
        <View style={styles.container}>
            <Text style={styles.title}>Gemma 2B Chatbot</Text>
            <TextInput
                style={styles.input}
                placeholder="Type your question..."
                value={input}
                onChangeText={setInput}
            />
            <Button title="Ask Gemma" onPress={handleRunGemma} />
            {output && <Text style={styles.output}>Response: {output}</Text>}
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#fff' },
    title: { fontSize: 18, fontWeight: 'bold', marginBottom: 10 },
    input: { width: '80%', height: 40, borderWidth: 1, marginBottom: 10, padding: 5 },
    output: { marginTop: 10, fontSize: 16, color: 'blue' },
});
```

---

## **🔹 Step 6: Run the App**
### **For Android**
```sh
npx react-native run-android
```

### **For iOS**
```sh
cd ios && pod install && cd ..
npx react-native run-ios
```

---

## **📌 Step 7: Optimize Performance**
### **1️⃣ Quantization for Faster Inference**
To **reduce size & speed up inference**, quantize the model:
```python
from onnxruntime.quantization import quantize_dynamic

quantized_model_path = "gemma-2b-quantized.onnx"
quantize_dynamic("gemma-2b.onnx", quantized_model_path)
```
Now use **`gemma-2b-quantized.onnx`** instead of `gemma-2b.onnx`.

---

## **🎯 Final Thoughts**
| Feature | Description |
|---------|------------|
| 📱 Supports | **Android & iOS** |
| ⚡ Speed | Faster with **ONNX runtime** |
| 🏗️ Framework | **React Native** |
| 🚀 Optimized | **Quantization & Mobile Inference** |
| 🔥 Model | **Gemma 2B (converted to ONNX)** |

Now **Gemma 2B runs locally** on mobile using ONNX Runtime in **React Native**! 🎯  

--- 


## Troubleshooting steps

It looks like **ONNX is not installed** in your Python environment. To fix this, follow these steps:

---

### **🔹 Step 1: Install ONNX and Dependencies**
Run the following command in your terminal:
```sh
pip install onnx onnxruntime onnxruntime-tools
```
If you need **GPU acceleration**, install the GPU version:
```sh
pip install onnxruntime-gpu
```

---

### **🔹 Step 2: Verify ONNX Installation**
Run this command in Python:
```python
import onnx
print(onnx.__version__)
```
If this prints a version number, **ONNX is installed correctly**.

---

### **🔹 Step 3: Install Additional Dependencies (If Needed)**
If you are using **Hugging Face Transformers and Optimum**, install:
```sh
pip install transformers optimum
```

---

### **🔹 Step 4: Retry Converting Gemma 2B to ONNX**
Now, run your **ONNX model conversion** script again:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import torch

model_name = "google/gemma-2b"

# Load the Gemma model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Export to ONNX
onnx_model_path = "gemma-2b.onnx"
ORTModelForCausalLM.from_pretrained(model, save_directory=onnx_model_path, export=True)

print(f"Gemma 2B converted to ONNX at {onnx_model_path}")
```

---

### **💡 If You Still Face Issues**
1. **Check Python Version**  
   Ensure you're using **Python 3.8+**:
   ```sh
   python --version
   ```

2. **Check Virtual Environment**  
   If you're using a **virtual environment**, activate it before installing:
   ```sh
   source venv/bin/activate  # For macOS/Linux
   venv\Scripts\activate      # For Windows
   ```

3. **Check `pip` Version**  
   ```sh
   python -m pip install --upgrade pip
   ```

---


It looks like the **Transformers** library is missing from your Python environment. You need to install it before running your script.

---

### **🔹 Step 1: Install `transformers`**
Run the following command in your terminal:
```sh
pip install transformers
```

If you're using ONNX, also install `optimum`:
```sh
pip install optimum
```

For **GPU acceleration**, install:
```sh
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
*(Make sure your CUDA version matches your PyTorch installation.)*

---

### **🔹 Step 2: Verify Installation**
After installation, check if `transformers` is available:
```python
import transformers
print(transformers.__version__)
```
If this prints a version number, **installation was successful**.

---

### **🔹 Step 3: Re-run Your ONNX Model Script**
Now, retry running your ONNX model conversion script.

---

### **💡 Still Facing Issues?**
1. **Check Python Version**
   ```sh
   python --version
   ```
   Make sure it's **Python 3.8+**.

2. **Check Virtual Environment**
   If you're using a virtual environment, activate it first:
   ```sh
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate      # Windows
   ```

3. **Upgrade `pip`**
   ```sh
   python -m pip install --upgrade pip
   ```
---

