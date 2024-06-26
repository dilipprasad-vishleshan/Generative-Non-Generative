from sentence_transformers import SentenceTransformer, util
import os

# Load the model from the saved directory
model = SentenceTransformer(os.path.join(os.getcwd(), 'saved_model'))

query1 = input('Enter query1: ')
query2 = input('Enter query2: ')

# Compute embeddings
embedding1 = model.encode(query1, convert_to_tensor=True)
embedding2 = model.encode(query2, convert_to_tensor=True)

# Compute cosine similarity
similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

# Define a threshold for determining relatedness
threshold = 0.7

# Print the result
if similarity > threshold:
    print("The queries are related.")
else:
    print("The queries are not related.")