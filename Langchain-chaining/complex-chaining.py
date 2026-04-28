from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

prompt1 = PromptTemplate(
    template="Make a detailed notes on this {topic}", input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Make 5 quizzes based on this {topic}", input_variables=["topic"]
)

prompt3 = PromptTemplate(
    template="from these notes and quizzes, make a single document. \n notes: {notes} \n quiz: {quiz} ",
    input_variables=["notes", "quiz"],
)

llm = HuggingFaceEndpoint(repo_id="google/gemma-4-31B-it", task="text-generation")
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

parallel_chain1 = RunnableParallel(
    {"notes": prompt1 | model | parser, "quiz": prompt2 | model | parser}
)

seperator_chain = RunnableParallel(
    {"notes&quiz": RunnablePassthrough(), "single_doc": prompt3 | model | parser}
)

chain = parallel_chain1 | seperator_chain

result = chain.invoke({"topic": "Agentic AI"})
print(result)

chain.get_graph().print_ascii()
