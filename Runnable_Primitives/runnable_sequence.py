from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema.runnable import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

prompt=PromptTemplate(
    template="You have to give me a joke about this {Topic}.",
    input_variables=["Topic"]
)

prompt2=PromptTemplate(
    template="you have to explain this {joke}",
    input_variables=["joke"]
)



parser=StrOutputParser()
chain=RunnableSequence(prompt,llm,parser,prompt2,llm,parser)

result=chain.invoke({"Topic": "AI"})

print("Result : \n",result)