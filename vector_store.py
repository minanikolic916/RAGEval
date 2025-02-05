from docs_loader import load_data
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, StorageContext

def init_vector_store(host, port, collection_name):
    client = qdrant_client.QdrantClient(
        host=host,
        port=port, 
    )
    if client.collection_exists(collection_name= collection_name):
        client.delete_collection(collection_name= collection_name)
    vector_store = QdrantVectorStore(client= client, collection_name= collection_name, enable_hybrid=True)
    return vector_store

def add_nodes_to_vec_store(nodes_path:str, collection_name:str):
    embed_model = HuggingFaceEmbedding(model_name = "BAAI/bge-m3")
    Settings.embed_model = embed_model
    vector_store = init_vector_store("localhost", 6333, collection_name= collection_name)
    nodes = load_data(folder_path=nodes_path)
    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding
    vector_store.add(nodes)
    return vector_store
  
def search_index(vector_store, query: str, top_k_dense:int, top_k_sparse:int):
    query_embedding = Settings.embed_model.get_query_embedding(query)
    #setting hybrid search
    query_mode = "hybrid"
    vec_store_q = VectorStoreQuery(
        query_embedding= query_embedding, similarity_top_k=top_k_dense, sparse_top_k= top_k_sparse, mode = query_mode
    )
    result = vector_store.query(vec_store_q)
    return result
