Here’s your complete **LLaMA 3.2 MCP CLI Agent** package:

**[Download llama_mcp_cli_agent.zip](sandbox:/mnt/data/llama_mcp_cli_agent.zip)**

Unzip it, install dependencies with `pip install -r requirements.txt`, and you’re good to go!

Let me know if you want:
- Docker support
- A Web UI
- More tools (e.g., web search, reminders, or weather)

Your **LLaMA 3.2 MCP CLI Agent** is ready!

### **What’s Included**
- Google Calendar scheduling
- Gmail email sending
- SQLite-based note saving
- Local CLI interface
- Runs with `llama3` model via Ollama

### **How to Run**
1. Install dependencies:
```bash
pip install langchain langchain-core langchain-community langchain-ollama google-api-python-client google-auth-httplib2 google-auth-oauthlib

pip install -r requirements.txt
```

2. Pull LLaMA 3.2 model in Ollama:
```bash
ollama pull llama3
```

3. Set up:
- Place `google_credentials.json` in the `credentials/` folder.
- Use a Gmail App Password and update `sender_email` and `sender_password` in `email_tool.py`.

4. Run the CLI:
```bash
python main.py
```

### **Example Commands**
- `Schedule project meeting on 2025-04-10T14:00 for 60 minutes`
- `Send an email to bob@example.com with subject 'Hello' and body 'Meeting at 3PM'`
- `Create a note titled 'Trip Ideas' with content 'Visit Japan in fall'`

---

# Examples

Here are some **natural language prompts** you can use to **automate tasks** with your LLaMA 3.2 MCP CLI agent. These work directly in the CLI — no commands or flags needed:

---

### 🗓️ **Google Calendar Prompts**
```text
Schedule a meeting with John on April 10th at 2pm for 90 minutes.
Book a call titled "Investor Review" at 2025-04-08T16:30 for 45 minutes.
Create an event: Team Sync on 2025-04-12T10:00, duration 30 minutes.
```

---

### 📧 **Gmail Email Prompts**
```text
Send an email to alice@example.com with subject "Trip Plans" and body "Let’s finalize the Japan itinerary."
Email bob@example.com with the subject 'Project Delay' saying 'We need to move the deadline by one week.'
Send an email to hr@company.com about 'Leave Application' saying 'I’ll be on vacation from April 15-20.'
```

---

### 📝 **Note-Taking Prompts**
```text
Create a note titled "Shopping List" with content "Eggs, Milk, Bread, Bananas".
Make a note called 'Meeting Summary' that says 'Discussed roadmap and Q2 hiring goals.'
Save a note titled 'Ideas for blog' with content 'LangChain use cases, local LLMs, offline agents.'
```

---

### ⚡ **Combo Ideas for Task Automation**
These will intelligently trigger multiple tools via MCP:
```text
Remind me to email the client tomorrow and also block 30 minutes on my calendar.
Create a note with my talking points and then schedule a prep session before the meeting.
Send Alice the itinerary and schedule a follow-up call for next Monday at 11am.
```

---

Would you like me to:
- Add voice input support?  
- Export these prompts as a `README` or test suite?  
- Enable memory or context-aware chaining?
