from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
chat_history = [
    SystemMessage("You are a hindi standup comedian. give all answers in funny way, in hindi language but in english text."),
    HumanMessage("who is PM of Pakistan?")
]

result = model.invoke(chat_history)
chat_history.append(AIMessage(content=result.content))

print(chat_history)