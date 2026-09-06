from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()


def GetTopic1(dic_of_Topics):
    return {"Topic":dic_of_Topics["Topic1"]}

def GetTopic2(dic_of_Topics):
    return {"Topic":dic_of_Topics["Topic2"]}

prompt1= PromptTemplate(
    template="You have to give me Short Notes on this {Topic}.",
    input_variables=["Topic"]
)


prompt2= PromptTemplate(
    template="You have to give me Short Notes on this {Topic}.",
    input_variables=["Topic"]
)


prompt3=PromptTemplate(
    template="From {Notes_For_Topic1} and {Notes_For_Topic2} generate a merged report for both.",
    input_variables=["Notes_For_Topic1","Notes_For_Topic2 "]
)

llm1=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7
)

llm2=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7
)

parser=StrOutputParser()


Parallel_Chain=RunnableParallel({
    "Notes_For_Topic1": GetTopic1 | prompt1 | llm1 | parser,
    "Notes_For_Topic2": GetTopic2 | prompt2 | llm2 | parser 
})

Chain_End=prompt3 | llm1 | parser

Merged_Chain=Parallel_Chain | Chain_End

result=Merged_Chain.invoke({
                            "Topic1": "Black Hole",
                            "Topic2":"Unemployment in Pakistan."
                            })

print("Result : \n",result)