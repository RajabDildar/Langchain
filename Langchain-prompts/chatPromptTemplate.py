from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

Chat_Template = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful {domain} expert'),
    # we can also use MessagePlaceholder for chat history for better context
    ('human', 'Explain this {topic} in easy way')
])

prompt = Chat_Template.invoke({"domain":"Psychology","topic":"procrastination"})

result = model.invoke(prompt)
print(result.content)