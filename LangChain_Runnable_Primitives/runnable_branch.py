from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence,RunnableBranch,RunnablePassthrough,RunnableLambda,RunnableParallel
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

prompt1=PromptTemplate(
    template="You have to generate a detailed report on this {Topic}.",
    input_variables=["Topic"]
)

prompt2=PromptTemplate(
    template="You have to summarize this report under 100 words: {report}.",
    input_variables=["report"]
)

parser =StrOutputParser()

report_generator=RunnableSequence(prompt1,llm,parser)

last_chain=RunnableParallel({
    "word_count": RunnableLambda(lambda x: len(x.split())),
    "report": RunnablePassthrough()
}
)

branch_chain =RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2,llm,parser,last_chain)),
    RunnablePassthrough()
)

final_chain=report_generator | branch_chain

print("Generating Report...")
result=final_chain.invoke({"Topic": "Artificial Intelligence"})
print("Result : \n", result)
