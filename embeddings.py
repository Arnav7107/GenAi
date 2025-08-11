from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Delhi is the capital of India"
# vector = embedding.embed_query(text)

# print(str(vector))

document = [

"Sachin Tendulkar, The God of Cricket whose bat spoke the language of records and inspiration.",
"Major Dhyan Chand, The hockey wizard who made the ball obey like magic, earning Olympic glory.",
"Milkha Singh,  The Flying Sikh who turned personal tragedy into a sprint of determination and pride.",
"P. V. Sindhu, The badminton queen who smashed barriers to bring home world and Olympic medals.",
"Viswanathan Anand, The chess grandmaster who made India a force on the 64 squares."

]

vector = embedding.embed_documents(document)
print(str(vector))