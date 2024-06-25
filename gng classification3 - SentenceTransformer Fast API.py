from transformers import SentenceTransformer, util
from fastapi import FastAPI, Path, Query
from fastapi.responses import JSONResponse

# create a server
app = FastAPI()

def load_model():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return model

def generate_embedding(model, query):
    embedding = model.encode(query, convert_to_tensor=True)
    return embedding

@app.get('/dependencycheck')
def dependency_check(query1: str, query2: str, threshold: float =   Query(description="Threshould should be b/w [0,1]", ge=0, le=1)):
    
    # Load the model
    model = load_model()
    
    # Generate embeddings
    embedding1 = generate_embedding(model, query1)
    embedding2 = generate_embedding(model, query2)
    
    # Compute cosine similarity
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

    # Display result
    if similarity > threshold:
        
        response_data = {
        'Similarity score': similarity,
        'Query Related Status': True
        }
        
        return JSONResponse(content=response_data)
    else:
        
        response_data = {
        'Similarity score': similarity,
        'Query Related Status': False
        }
        
        return JSONResponse(content=response_data)