from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

get_detailed_report_prompt = PromptTemplate(
    template="write a detailed report on {topic}", input_variables=["topic"]
)
get_key_points_prompt = PromptTemplate(
    template="write 5 key notes of this report: {report}", input_variables=["report"]
)

llm = HuggingFaceEndpoint(repo_id="google/gemma-4-31B-it", task="text-generation")
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

chain = (
    get_detailed_report_prompt | model | parser | get_key_points_prompt | model | parser
)
result = chain.invoke({"topic": "Evolution of AI in 2025"})
print(result)

chain.get_graph().print_ascii()
