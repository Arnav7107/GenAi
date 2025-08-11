from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is the capital of India"
# vector = embedding.embed_query(text)

# print(str(vector))

document = [

"Sachin Tendulkar, The God of Cricket whose bat spoke the language of records and inspiration.",
"Major Dhyan Chand, The hockey wizard who made the ball obey like magic, earning Olympic glory.",
"Milkha Singh,  The Flying Sikh who turned personal tragedy into a sprint of determination and pride.",
"P. V. Sindhu, The badminton queen who smashed barriers to bring home world and Olympic medals.",
"Viswanathan Anand, The chess grandmaster who made India a force on the 64 squares.",
"Yuvraj Singh, The fearless cricketer who battled cancer and still hit six sixes in an over into history."

]

query = "Tell me about cricketer Singh."

doc_embeddings = embedding.embed_documents(document)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

print(scores)

index, score = sorted(list(enumerate(scores)), key = lambda x: x[1])[-1]

print(document[index])
print("Similarity Score is: ", score)

