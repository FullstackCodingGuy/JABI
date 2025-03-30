### **Setting Up ONNX Runtime in React Native for Mobile AI Inference**  
ONNX Runtime allows you to run **AI models on mobile devices** (Android & iOS) efficiently.  

This guide will walk you through **installing and using ONNX Runtime in React Native**.

---

## **📌 Step 1: Install ONNX Runtime for React Native**  

1️⃣ **Initialize a React Native Project (if not created yet)**  
```sh
npx react-native init OnnxExample
cd OnnxExample
```

2️⃣ **Install ONNX Runtime for React Native**  
```sh
npm install onnxruntime-react-native
```

3️⃣ **Link Native Modules** (For older React Native versions `<0.60`)  
```sh
npx react-native link onnxruntime-react-native
```

---

## **📌 Step 2: Add the ONNX Model to Your Project**
1️⃣ Place your ONNX model (`model.onnx`) in your React Native project under:  
```
/assets/model.onnx
```

2️⃣ For **iOS**, update `Info.plist` to allow asset access:
```xml
<key>UIFileSharingEnabled</key>
<true/>
<key>LSSupportsOpeningDocumentsInPlace</key>
<true/>
```

3️⃣ For **Android**, add `assets` configuration in `android/app/src/main/assets`.

---

## **📌 Step 3: Run an ONNX Model in React Native**
Here’s a simple example of **loading and running an ONNX model** in React Native:

### **Create `onnx-runner.js`**
```javascript
import { OrtSession, Tensor } from 'onnxruntime-react-native';
import { Platform, NativeModules } from 'react-native';

// Helper function to load the ONNX model
export async function loadModel() {
    const modelPath = Platform.OS === 'ios'
        ? `${NativeModules.NativeUtils.getDocumentDirectory()}/model.onnx`
        : 'file:///android_asset/model.onnx';

    const session = await OrtSession.create(modelPath);
    return session;
}

// Run inference with the ONNX model
export async function runModel(session, inputArray) {
    try {
        const inputTensor = new Tensor('float32', new Float32Array(inputArray), [1, inputArray.length]);
        const feeds = { input: inputTensor };
        const results = await session.run(feeds);

        console.log("ONNX Model Output:", results.output.data);
        return results.output.data;
    } catch (error) {
        console.error("ONNX Inference Error:", error);
        return null;
    }
}
```

---

## **📌 Step 4: Use ONNX Model in React Native Component**
Now, modify `App.js` to **load and run the ONNX model**.

### **Update `App.js`**
```javascript
import React, { useEffect, useState } from 'react';
import { View, Text, Button, StyleSheet } from 'react-native';
import { loadModel, runModel } from './onnx-runner';

export default function App() {
    const [model, setModel] = useState(null);
    const [output, setOutput] = useState(null);

    useEffect(() => {
        async function initialize() {
            const session = await loadModel();
            setModel(session);
        }
        initialize();
    }, []);

    const handleRunModel = async () => {
        if (!model) return;
        const result = await runModel(model, [1.0, 2.0, 3.0, 4.0]); // Example input
        setOutput(result);
    };

    return (
        <View style={styles.container}>
            <Text style={styles.title}>ONNX React Native Example</Text>
            <Button title="Run Model" onPress={handleRunModel} />
            {output && <Text style={styles.output}>Output: {JSON.stringify(output)}</Text>}
        </View>
    );
}

const styles = StyleSheet.create({
    container: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#fff' },
    title: { fontSize: 18, fontWeight: 'bold', marginBottom: 10 },
    output: { marginTop: 10, fontSize: 16, color: 'blue' },
});
```

---

## **📌 Step 5: Run the App**
### **For Android**
1️⃣ Ensure you have a physical device or emulator running  
2️⃣ Run:
```sh
npx react-native run-android
```

### **For iOS**
1️⃣ Install dependencies for iOS:
```sh
cd ios && pod install && cd ..
```
2️⃣ Run:
```sh
npx react-native run-ios
```

---

## **📌 Step 6: Optimize Model for Mobile**
To reduce size and improve performance, **quantize** the ONNX model:
```python
from onnxruntime.quantization import quantize_dynamic

quantized_model_path = "model_quantized.onnx"
quantize_dynamic("model.onnx", quantized_model_path)
```
Then, use `model_quantized.onnx` instead of `model.onnx`.

---

## **🎯 Final Thoughts**
| **Feature** | **Description** |
|------------|----------------|
| 📱 Supports | Android & iOS |
| ⚡ Optimized | Quantization + ONNX runtime |
| 🏗️ Framework | React Native |
| 🚀 Speed | Fast inference on mobile |

This setup enables **AI-powered mobile applications** with ONNX in **React Native**. 🚀  