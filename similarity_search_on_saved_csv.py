import pandas as pd
from sentence_transformers import SentenceTransformer, util
import os

# Load the model from the saved directory
model = SentenceTransformer(os.path.join(os.getcwd(), 'saved_model'))

# Read .csv file
df = pd.read_csv('generalized.csv')

def sim_search(user_query):
    # Compute embedding for the user query
    user_embedding = model.encode(user_query, convert_to_tensor=True)
    
    # Compute embeddings for all saved queries
    saved_queries = df['Query'].tolist()
    saved_embeddings = model.encode(saved_queries, convert_to_tensor=True)
    
    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(user_embedding, saved_embeddings)[0]
    
    # Create a DataFrame with results
    result_df = pd.DataFrame({
        'Saved_Query': saved_queries,
        'User_Query': [user_query] * len(saved_queries),
        'Similarity_Score': similarities.cpu().numpy()
    })
    
    return result_df

# User input
user_query = input("Enter your query: ")

# Associate a name to dataframe object
result_df = sim_search(user_query)
result_df.to_csv('userVSsavedQuery_similarity_info.csv')
print(result_df)
