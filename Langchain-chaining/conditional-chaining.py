from typing import Literal

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(repo_id="google/gemma-4-31B-it", task="text-generation")
model = ChatHuggingFace(llm=llm)


class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Give sentiment of the feedback"
    )


parser1 = PydanticOutputParser(pydantic_object=Feedback)
parser2 = StrOutputParser()


prompt1 = PromptTemplate(
    template="Give sentiment of this feedback: \n {feedback}. \n {format_output}",
    input_variables=["feedback"],
    partial_variables={"format_output": parser1.get_format_instructions()},
)

prompt2 = PromptTemplate(
    template="Give a 4-5 lines response to user according to this positive feedback: \n {sentiment}.",
    input_variables=["sentiment"],
)

prompt3 = PromptTemplate(
    template="Give a 4-5 lines response to user according to this negative feedback: \n {sentiment}.",
    input_variables=["sentiment"],
)

classifier_chain = prompt1 | model | parser1

response_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt2 | model | parser2),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser2),
    RunnableLambda(lambda x: "could not find sentiment"),
)

feedback_chain = classifier_chain | response_chain
result = feedback_chain.invoke({"feedback": "This is a worst phone i have ever had."})
print(result)

feedback_chain.get_graph().print_ascii()
