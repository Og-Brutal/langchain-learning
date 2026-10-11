from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema.runnable import RunnableParallel,RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

prompt1=PromptTemplate(
    template="you have genenrate a textual tweet on this {Topic}.",
    input_variables=["Topic"]
)

prompt2=PromptTemplate(
    template="you have to generate a textual linkedin post on this {Topic}.",
    input_variables=["Topic"]
)

parser=StrOutputParser()

Parallel_Chain=RunnableParallel({
    "Tweet": RunnableSequence(prompt1,llm,parser),
    "Linkedin":RunnableSequence(prompt2,llm,parser)
})

result=Parallel_Chain.invoke({"Topic": "AI"})

print("Result : \n",result)

print("\nTweet : \n",result["Tweet"])
print("\nLinkedin : \n",result["Linkedin"])