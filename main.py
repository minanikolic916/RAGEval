import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#local file imports 
from inference import load_model_and_tokenizer, load_pipeline, get_model_response
from vector_store import add_nodes_to_vec_store, search_index
from retrieval import get_nodes_with_scores, similarity_cutoff_nodes, rerank_nodes_colbert
from utils import final_display_context
from log_data.log_utils import log_data_to_json

app = FastAPI()

#enable CORS
allowed_origins = [
    "http://localhost:4200"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["X-Requested-With", "Content-Type"],
)

FILE_PATH = "./data_without_questions"
LOGGING_PATH = f"./log_data/log_runs/RAG_LOG.json"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
#MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
#MODEL_NAME = "google/gemma-2-9b-it"

class QuestionRequest(BaseModel):
    question:str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.on_event("startup")
async def create_index():
    try:
        app.state.vec_store = add_nodes_to_vec_store(nodes_path=FILE_PATH, collection_name="pirot_kolekcija")
        print(f"Cvorovi uspesno dodati u bazu")
    except Exception as e:
        print("Dodavanje cvorova u vektosku bazu je neuspesno")
        raise HTTPException(status_code=500, detail="Dodavanje cvorova neuspesno")

@app.on_event("startup")
async def load_model():
    model_name = MODEL_NAME
    try:
        model, tokenizer = load_model_and_tokenizer(model_name)
        app.state.pipe = load_pipeline(model = model, tokenizer= tokenizer)
        print(f"Model: {model_name} uspesno ucitan")

    except Exception as e:
        print("Model i tokenizator nisu ucitani: {e}")
        raise HTTPException(status_code=500, detail="Greska prilikom ucitavanja modela")


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    question = request.question
    semantic_search_results = search_index(app.state.vec_store, query=question, top_k_dense=3, top_k_sparse=5)
    cutoff_nodes = similarity_cutoff_nodes(result= semantic_search_results, query=question, top_k=3, score = 0.55)
    #nodes_colbert_rerank = rerank_nodes_colbert(cutoff_nodes, query= question, top_k= 2)

    if cutoff_nodes:
        context = final_display_context(cutoff_nodes)
        print(context)
    else:
        return f"Žao mi je, ne razumem Vaše pitanje. Molim Vas da preformulišete."

    try:
        result = get_model_response(pipe = app.state.pipe, context=context, question=question)
        log_data_to_json(LOGGING_PATH, question = question, context = context, answer = result)
        print(f"Logging done.")
        return result 
    except Exception as e:
        raise HTTPException(status_code=500, detail="Problem prilikom procesiranja pitanja")