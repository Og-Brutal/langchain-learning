from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


llm=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7
)

prompt= PromptTemplate(
    template="Hey generate five interesting facts about {Topic}",
    input_variables=["Topic"]
)

parser=StrOutputParser()


chain = prompt | llm | parser

result= chain.invoke({"Topic":"Python programming language"})

print("Unstructured Output : \n",result)