from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

embedding_vector = embedding_model.embed_query("What is the capital of Pakistan?")

print("Embedding Vector : \n\n", embedding_vector)
