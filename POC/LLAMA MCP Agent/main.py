from langchain_ollama import OllamaLLM as Ollama
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from tools.calendar_tool import create_event
from tools.email_tool import send_email
from tools.note_tool import create_note

llm = Ollama(model="llama3.2:3b")

tools = [create_event, send_email, create_note]

tool_names = [tool.name for tool in tools]
tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

PREFIX = "You are an assistant that helps with task automation."
SUFFIX = "Provide step-by-step reasoning and actions."
FORMAT_INSTRUCTIONS = "Use the tools provided to complete the tasks."


full_prompt_template = f"""{PREFIX}

{tool_descriptions}

{FORMAT_INSTRUCTIONS}

{SUFFIX}

Question: {{input}}
Thought: Let's think step by step.
{{agent_scratchpad}}
"""


# Prompt MUST contain {input}, {agent_scratchpad}, {tools}, and {tool_names}
prompt_template = """
You are an assistant that helps with task automation like managing calendar events, sending emails, and taking notes.

You have access to the following tools:
{tools}

The names of the tools are:
{tool_names}

When given a user's request, decide what to do step by step using the tools.

Begin!

Question: {input}
Thought: Let's think step by step.
{agent_scratchpad}

Action: create_event
Action Input: {"title": "Team Sync", "date": "2025-04-08", "time": "10:30"}

"""

# prompt = PromptTemplate(
#     input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
#     template=prompt_template
# )

prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    partial_variables={"tool_names": ", ".join(tool_names), "tools": tool_descriptions},
    template=full_prompt_template,
)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

if __name__ == "__main__":
    print("Welcome to LLaMA MCP Agent CLI")
    while True:
        try:
            user_input = input("\nAsk: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            result = agent_executor.invoke({"input": user_input})
            print("\nResult:", result["output"])
        except Exception as e:
            print("Error:", e)
