from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain.agents import create_agent

# Import the tools from tools.py
from tools import tools

load_dotenv()

# 1. Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)


# 2. parser
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# 3. Create the agent
# Pass your instructions as a simple string or SystemMessage
system_instructions = "You are a research assistant. Help generate research papers."

agent = create_agent(
    model=llm,
    tools=tools,  # Add tools here when ready
    system_prompt=system_instructions,
    response_format=ResearchResponse,  # Modern way to handle structured output
)

# 4. Invoke the agent
# Modern agents use a 'messages' list for input by default
result = agent.invoke({"messages": [("user", "What is the impact of AI on society?")]})

# Access the structured output
print(result["structured_response"])
