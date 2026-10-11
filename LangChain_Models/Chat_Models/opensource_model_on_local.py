from langchain_huggingface import HuggingFacePipeline,ChatHuggingFace

import os 

os.environ["HF_HOME"]="E:/Random_Projects/HUGGINGFACE_CACHE"  # Set the cache directory for Hugging Face models

LLM=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation"
    )

chatModel=ChatHuggingFace(llm=LLM)

result=chatModel.invoke("What is capital of France?")

print("Result : \n\n", result.content)



