# Example Prompts:
# What is the weather in Paris?
# Add 10, 20, and 30 for me.
# What time is it now?

# The model heavily depends on tool docstrings for choosing the right tool. Here's how to make them ReAct-agent-friendly:


from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM  
import datetime
import warnings
warnings.filterwarnings("ignore", message='fields may not start with an underscore', category=RuntimeWarning)

# === Step 1: Define Custom Tools ===

@tool
def get_weather(location: str) -> str:
    """Use this tool ONLY when the user asks about the weather in a location. Input is a location string."""

    return f"The weather in {location} is sunny and 25°C."

@tool
def calculate_sum(numbers: str) -> str:
    """Use this tool ONLY when the user wants to add numbers. Input must be comma-separated values like '1, 2, 3'."""

    try:
        nums = [float(n.strip()) for n in numbers.split(",")]
        return f"The sum is {sum(nums)}."
    except:
        return "Invalid input. Please provide comma-separated numbers."

@tool
def get_time(_: str = "") -> str:
    """Use this tool ONLY when the user asks for the current time."""
    return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."

tools = [get_weather, calculate_sum]

# === Step 2: Load LLaMA via Ollama ===

# llm = OllamaLLM(model="gemma:2b")
llm = OllamaLLM(model="llama3.2:3b")

# === Step 3: Initialize Agent ===

# Use initialize_agent’s agent_kwargs to inject context into the reasoning. 

agent = initialize_agent(
    tools,
    llm,
    # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    # max_iterations=5,
    agent_kwargs={
        "system_message": (
            "You are a helpful assistant. Choose the most relevant tool based on user intent. "
            "Use the weather tool only for weather questions, the sum tool for adding numbers, and the time tool for current time."
        )
    }
)

# === Step 4: Run User Query ===

if __name__ == "__main__":
    print("🔧 Simple Tool Agent (Type 'exit' to quit)")
    while True:
        user_input = input("\nAsk something: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            response = agent.invoke(user_input)  # ✅ Updated to invoke
            print(f"\n🧠 Agent: {response}")
        except Exception as e:
            print(f"⚠️ Error: {e}")

