from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.tools import tool

from langchain_core.messages import HumanMessage

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="google/gemma-4-31B-it", task="text-generation")
model = ChatHuggingFace(llm=llm)


@tool
def Multiply(a: int, b: int) -> int:
    """This tool multiplies two numbers"""
    return a * b


# tool binding -> making tools available to llms/models
model_with_tools = model.bind_tools([Multiply])

# result = model_with_tools.invoke("Hi!") # for these normal things, model will not see or use tools
# print(result)

query = HumanMessage(content="can you multiply 2 and 15")
messages = [query]
result = model_with_tools.invoke(
    messages
)  # for these things, model will suggest which tools to call and what is the input for the tool.
# print(result)
messages.append(result)

tool_msg = Multiply.invoke(result.tool_calls[0])
messages.append(tool_msg)
# print(messages)

res = model.invoke(messages)
print(f"res is this: {res.content}")
