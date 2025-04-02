# Agentic AI example using LangChain with LLaMA 3.2 via Ollama.
## Wikipedia Search

Here's an **Agentic AI example** using **LangChain** with **LLaMA 3.2** via **Ollama**.  

## **✅ Features**  
- Uses **LangChain** with **LLaMA 3.2** (Ollama).  
- Creates an **Agent** with **Tools** (e.g., Wikipedia search).  
- Uses **Memory** to retain context.  

---

## **🔹 Install Dependencies**  
Run the following in your terminal:  
```sh
pip install langchain langchain-community langchain-openai ollama wikipedia
```

---

## **🔹 Step 1: Setup LangChain with LLaMA 3.2 (Ollama)**
```python
from langchain_community.llms import Ollama

# Load LLaMA 3.2 model via Ollama
llm = Ollama(model="meta/llama3.2")

# Test response
print(llm.invoke("What is Agentic AI?"))
```
✅ **If this works, your LLaMA 3.2 model is ready!**  

---

## **🔹 Step 2: Create an Agent with Memory & Tools**
```python
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_community.llms import Ollama
import wikipedia

# Initialize LLaMA 3.2 model
llm = Ollama(model="meta/llama3.2")

# Define a Wikipedia search tool
def search_wikipedia(query: str):
    return wikipedia.summary(query, sentences=2)

wiki_tool = Tool(
    name="Wikipedia Search",
    func=search_wikipedia,
    description="Searches Wikipedia for information"
)

# Set up memory to retain context
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent
agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=[wiki_tool],
    llm=llm,
    memory=memory,
    verbose=True
)

# Test the agent
response = agent.invoke("Tell me about artificial intelligence and find more details from Wikipedia.")
print(response)
```
✅ **This agent will use Wikipedia and memory while responding!**  

---

## **🔹 Step 3: Add More Tools (e.g., Sending Emails, Setting Meetings)**
You can extend the agent by integrating **task automation tools**, such as **sending emails, setting up meetings, or controlling APIs**.

Example: **Email Sending Tool**
```python
import smtplib
from email.mime.text import MIMEText

def send_email(subject: str, body: str, to: str):
    sender = "your-email@example.com"
    password = "your-app-password"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, to, msg.as_string())
    
    return f"Email sent to {to}!"

email_tool = Tool(
    name="Send Email",
    func=lambda text: send_email("AI Notification", text, "recipient@example.com"),
    description="Sends an email with the provided text."
)

# Add to Agent
agent.tools.append(email_tool)

# Test sending an email
response = agent.invoke("Send an email saying 'Meeting at 10 AM' to recipient@example.com")
print(response)
```
✅ **Now your AI Agent can send emails!**  

---

## **🚀 Summary**
- ✅ **Used LLaMA 3.2 (Ollama) in LangChain.**  
- ✅ **Created an Agent with Wikipedia search.**  
- ✅ **Added memory to retain conversation history.**  
- ✅ **Extended with automation tools (e.g., Email Sending).**  

Would you like to add **WhatsApp integration** or **more task automation?** 🚀