from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()  # Load environment variables from .env file


llm=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=1.7
)

class Schema(BaseModel):
   name:str = Field(..., description="Name of the person")
   age:int = Field(...,gt=18, description="Age of the person")
   city:str = Field(..., description="City of the person")


parser=PydanticOutputParser(pydantic_object=Schema)

template= PromptTemplate(
    template="Hey you have to give me name, age and city of a fictional person living in {State} \n\n {Format_instructions}",
    input_variables=["State"],
    partial_variables={"Format_instructions": parser.get_format_instructions()}
)

chain = template | llm | parser

result = chain.invoke({"State":"California"})

print("Structured Output : \n\n",result)