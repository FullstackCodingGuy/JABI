import ollama

# MODELNAME="mistral"
# MODELNAME = "llama3.2"      # latest
MODELNAME = "llama3.2:3b"     # faster
# MODELNAME = "gemma:2b"
# MODELNAME = "codellama"     # slower
# MODELNAME = "qwen2.5"
# MODELNAME = "deepseek-r1:7b"  # slowest

class AIAgent:
    def __init__(self, model=MODELNAME):
        self.model = model
        self.history = []  # Memory for chat context
        print("Agent Init Done. Model:", self.model)

    def chat(self, user_input):
        # Add conversation history for context
        prompt = "\n".join(self.history) + f"\nUser: {user_input}\nAI:"
        response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}])

        ai_response = response["message"]["content"]
        self.history.append(f"User: {user_input}")
        self.history.append(f"AI: {ai_response}")

        return ai_response

# Run the AI agent in the terminal
agent = AIAgent()
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    print("AI Agent:", agent.chat(user_input))
