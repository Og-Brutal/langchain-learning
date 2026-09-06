from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file



temp1=PromptTemplate(
    template="Hey give me  a detailed report on this {Topic}.",
    input_variables=["Topic"]
)

temp2=PromptTemplate(
    template="Hey generate five interesting facts about {Text}",
    input_variables=["Text"]
)

llm=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7
)

parser=StrOutputParser()


chain= temp1 | llm | parser | temp2 | llm | parser

result=chain.invoke({"Topic":"Unemployement in Pakistan !"})

print("Result : \n",result)