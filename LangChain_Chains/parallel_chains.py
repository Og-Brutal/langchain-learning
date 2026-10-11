from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableParallel
from dotenv import load_dotenv

load_dotenv()




prompt1= PromptTemplate(
    template="You have to give me Short Notes on this {Topic}.",
    input_variables=["Topic"]
)


prompt2= PromptTemplate(
    template="You have to give me a Short quiz on this {Topic}.",
    input_variables=["Topic"]
)


prompt3=PromptTemplate(
    template="From {Notes} and {Quiz} make merged one single report of both having Notes and quiz both.",
    input_variables=["Notes","Quiz "]
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
    "Notes": prompt1 | llm1 | parser,
    "Quiz":prompt2 | llm2 | parser 
})

Chain_End=prompt3 | llm1 | parser

Merged_Chain=Parallel_Chain | Chain_End

result=Merged_Chain.invoke({"Topic": "Black Hole"})

print("Result : \n",result)