from sentence_transformers import SentenceTransformer, util
import os

def create_similarity_search_model(model_name):
    # Load the model
    model2 = SentenceTransformer(model_name)
    # Save the model to a directory
    save_path = os.path.join(os.getcwd(), 'saved_model')
    model2.save(save_path)

# Create and save the model
create_similarity_search_model("paraphrase-MiniLM-L6-v2")
