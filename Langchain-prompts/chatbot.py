from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
chat_history = [
    SystemMessage("You are a hindi standup comedian. give all answers in funny way, in hindi language but in english text.")
]

while True:
    prompt = input("You: ")
    chat_history.append(HumanMessage(content=prompt))
    if prompt == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)

# for dynamic prompts in chatmodels, we use ChatPromptTemplate