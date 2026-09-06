from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7
)



template1= PromptTemplate(
    template="You are a helpful assistant. give me a detailed report about this {topic}.",
    input_variables=["topic"]
)

template2= PromptTemplate(
    template="describe this report in 5 lines {text}.",
    input_variables=["text"]
)

def print_input(input):
    print("Input : \n\n", input)
    return input
parser = StrOutputParser()

chain =template1  | llm  | parser  | template2 | llm  |parser 

result=chain.invoke({"topic":"Samsung Galaxy S24 Ultra"})



print("Result : \n\n", result)


#