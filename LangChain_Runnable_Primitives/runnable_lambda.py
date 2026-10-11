from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableLambda,RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

def word_count(text: str) -> int:
    return len(text.split())

llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

prompt=PromptTemplate(
    template="You have to give me a short joke about this {Topic}.",
    input_variables=["Topic"]
)

parser=StrOutputParser()

joke_generator=RunnableSequence(prompt,llm,parser)

parallel_chain=RunnableParallel({
    "Joke": RunnablePassthrough(),
    "WordCount": RunnableLambda(word_count)
})

final_chain=joke_generator | parallel_chain

result=final_chain.invoke({"Topic": "AI"})
print("Generating Joke...")
print("Result : \n", result)
