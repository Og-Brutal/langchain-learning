from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "The capital of Pakistan is Islamabad.",
    "Islamabad is the capital city of Pakistan.",
    "Lahore is famous for its Mughal-era architecture and food street.",
    "Python is a popular programming language for data science and AI.",
    "Machine learning models can learn patterns from data automatically.",
    "Deep learning is a subset of machine learning based on neural networks.",
    "Cricket is the most popular sport in Pakistan.",
    "Pakistan won the Champions Trophy in cricket history.",
    "The stock market fluctuates based on economic indicators.",
    "Bitcoin is a decentralized digital currency.",
    "Global warming is causing significant changes in climate patterns.",
    "Renewable energy sources like solar and wind reduce carbon emissions."
]

embedding_vectors = embedding_model.embed_documents(documents)

embedding_query = embedding_model.embed_query("who won the Champions Trophy?")

scores=cosine_similarity([embedding_query], embedding_vectors)[0]

index,score=sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]
print("Similarity Score : \n\n", score)
print("Most Similar Document Index : \n\n", index)
print("Most Similar Document : \n\n", documents[index])
