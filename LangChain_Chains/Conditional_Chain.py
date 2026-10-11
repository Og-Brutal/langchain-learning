from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv
from langchain_classic.schema.runnable import RunnableBranch,RunnableLambda

review="Mobile is not good."
load_dotenv()
parser=StrOutputParser()

class Classification(BaseModel):
    sentiment: Literal["Positive","Negative"] = Field(...,description="Description Sentiment of the review and it should positivie or negative.")


llm=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7
)
structured_llm=llm.with_structured_output(Classification)

temp1=PromptTemplate(
    template="You have to give me sentiment(postive,negative) of this {review}",
    input_variables=["review"]
)
temp2= PromptTemplate(
    template="you have to give a reposnse to this postive feedback {review}",
    input_variables=["review"]
)

temp3= PromptTemplate(
    template="you have to give a reposnse to this negative feedback {review}",
    input_variables=["review"]
)
classification_chain = temp1 | structured_llm 

def get_review(input):
    print("Hey input : ",input)
    return review


conditional_chain=RunnableBranch(
    (lambda x:x.sentiment=="Positive",get_review | temp2 | llm | parser),
    (lambda x:x.sentiment=="Negative",get_review | temp3 | llm | parser),
    RunnableLambda(lambda x : "out of scope !")
)

merged_chain=classification_chain | conditional_chain

result=merged_chain.invoke({review})

print(result)