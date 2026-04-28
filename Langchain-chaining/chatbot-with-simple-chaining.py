from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template="Give 5 interesting facts about {topic}", input_variables=["topic"]
)

llm = HuggingFaceEndpoint(repo_id="google/gemma-4-31B-it", task="text-generation")
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"topic": "cricket"})
print(result)

chain.get_graph().print_ascii()
