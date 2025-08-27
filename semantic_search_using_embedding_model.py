from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
warnings.simplefilter("ignore", category=FutureWarning)


load_dotenv()

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')


documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Tell me about Virat Kohli"

doc_embeddings = embedding.embed_documents(documents)
query_embeddings=  embedding.embed_query(query)

# print(cosine_similarity([query_embeddings], doc_embeddings))

scores = cosine_similarity([query_embeddings], doc_embeddings)

#print(list(enumerate(score)))
#print(sorted(list(enumerate(scores)), key = lambda x:x[1]))

print(sorted(list(enumerate(scores)), key = lambda x:x[1])[-1])

