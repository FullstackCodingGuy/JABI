# main.py

# Ask: Schedule a meeting titled "Project Kickoff" on 2025-04-10 at 3:00 PM.

from langchain_ollama import OllamaLLM as Ollama
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from pydantic import BaseModel, Field
import json

# --- Define Pydantic input schema ---
class EventInput(BaseModel):
    title: str = Field(..., description="Title of the event")
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    time: str = Field(..., description="Time in HH:MM AM/PM format")
    # location: str = Field(default="", description="Location of the event")
    # description: str = Field(default="", description="Event description")

# --- Define the tool function ---
def create_event_fn(title: str, date: str, time: str) -> str:
    print("------------------------------------------------")
    print("Creating event with details:")
    print(f"Title: {title}, Date: {date}, Time: {time}")
    return f"[SUCCESS] Event '{title}' created for {date} at {time}."

# --- Register the tool with proper schema ---
create_event = Tool.from_function(
    name="create_event",
    description="Schedules a calendar event. Provide title, date, and time.",
    func=create_event_fn,
    args_schema=EventInput,
)

tools = [create_event]

# --- Extract tool metadata for prompt ---
tool_names = [tool.name for tool in tools]
tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

# --- ReAct Prompt Template ---
prompt_template = """
You are an assistant that helps schedule calendar events.

You have access to the following tools:
{tools}

The names of the tools are:
{tool_names}

You MUST call tools with input as a valid JSON object. Do NOT wrap the JSON inside a string. Just pass the object with the required fields.

Use this format:

Thought: I need to schedule an event.
Action: create_event
Action Input: {{
  "title": "Project Kickoff",
  "date": "2025-04-10",
  "time": "3:00 PM"
}}

Observation: ✅ Event 'Project Kickoff' created.

Thought: The task is complete.
Final Answer: The event has been scheduled successfully.

Begin!

Question: {input}
Thought: Let's think step by step.
{agent_scratchpad}
"""


# --- Prompt setup ---
prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    partial_variables={"tool_names": ", ".join(tool_names), "tools": tool_descriptions},
    template=prompt_template,
)

# --- LLM setup ---
llm = Ollama(model="llama3.2:3b")

# --- Create agent ---
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    handle_parsing_errors=True
)

# --- CLI ---
if __name__ == "__main__":
    print("✅ Welcome to the Calendar Event Scheduler CLI (powered by LLaMA 3.2 + LangChain)\n")
    while True:
        try:
            user_input = input("Ask: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            result = agent_executor.invoke({"input": user_input})
            print("\n🎉 Result:", result["output"])
        except Exception as e:
            print("❌ Error:", e)
