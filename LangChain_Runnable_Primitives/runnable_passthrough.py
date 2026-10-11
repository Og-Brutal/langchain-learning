from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.schema.runnable import RunnableParallel,RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

prompt1=PromptTemplate(
    template="you have to generate a joke on this {Topic}.",
    input_variables=["Topic"]
)

prompt2=PromptTemplate(
    template="you have explain this {joke}.",
    input_variables=["joke"]
)
parser=StrOutputParser()

joke_generate= RunnableSequence(prompt1,llm,parser)

parallel_chain=RunnableParallel({
    "Joke": RunnablePassthrough(),
    "Explanation": RunnableSequence(prompt2,llm,parser)
})

final_chain=joke_generate | parallel_chain

result=final_chain.invoke({"Topic": "AI"})
print("Generating Joke...")
print("Result : \n", result)

# ================= RunnablePassthrough (Revision Notes) =================
#
# WHAT:
#   Passes its input through UNCHANGED (same as RunnableLambda(lambda x: x)).
#
# WHY:
#   In a chain, each step's output REPLACES the previous data.
#   If a later step needs the ORIGINAL input too, passthrough keeps it alive.
#
# 1) Plain RunnablePassthrough()
#   - Forwards the input as-is.
#   - Typical use: RAG, inside RunnableParallel
#       RunnableParallel(
#           context=retriever,                 # input -> docs
#           question=RunnablePassthrough(),    # input -> unchanged
#       ) | prompt | llm | parser
#
# 2) RunnablePassthrough.assign(new_key=runnable)
#   - Input MUST be a dict.
#   - Keeps all existing keys AND adds new computed key(s).
#       RunnablePassthrough.assign(notes=notes_chain)
#       .invoke({"topic": "x"})  ->  {"topic": "x", "notes": "..."}
#   - Chain several: later steps can read earlier keys (modern replacement
#     for SequentialChain).
#   - Multiple keys in one assign() run in PARALLEL.
#
# WHEN TO USE:
#   - A later step needs both the original input and a computed result
#   - Same input must go to a retriever AND the prompt (RAG)
#   - Accumulating values in a dict across steps (.assign)
#
# WHEN NOT:
#   - Simple prompt | llm | parser (nothing to carry forward)
#
# VS RunnableLambda:
#   - Passthrough/assign = keep or add data (less code, supports async,
#     streaming, tracing, doesn't mutate the input dict)
#   - RunnableLambda = transform data in a custom way
#
# ONE-LINER: Passthrough keeps the original input alive while other
#            steps transform it.
# =========================================================================