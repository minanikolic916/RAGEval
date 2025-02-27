from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from vector_store import search_index
from typing import Optional

def get_nodes_with_scores(result, query:str, top_k: int):
    #result = search_index(query= query, top_k=top_k)
    nodes_with_scores = []
    for index, node in enumerate(result.nodes):
        score : Optional[float] = None
        if result.similarities is not None:
            score = result.similarities[index]
        nodes_with_scores.append(NodeWithScore(node=node, score=score))
    return nodes_with_scores

def similarity_cutoff_nodes(result, query:str, score:float, top_k:int):
    processor = SimilarityPostprocessor(similarity_cutoff=score)
    nodes = get_nodes_with_scores(result, query=query, top_k=top_k)
    cutoff_nodes = processor.postprocess_nodes(nodes)
    return cutoff_nodes

def rerank_nodes_sent_transformers(result, query:str, top_k:int, score:float):
    rerank_model = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=top_k
    )
    #nodes = get_nodes_with_scores(result, query=query, top_k=top_k)
    #nodes = similarity_cutoff_nodes(result = result, query=query, score=score, top_k=top_k)
    #reranked_nodes = rerank_model.postprocess_nodes(nodes, query_str=query)
    reranked_nodes = rerank_model.postprocess_nodes(nodes, query_str=query)
    return reranked_nodes

def rerank_nodes_colbert(result, query:str, top_k:int):
    colbert_reranker = ColbertRerank(
        top_n=top_k, 
        model="colbert-ir/colbertv2.0",
        tokenizer="colbert-ir/colbertv2.0",
        keep_retrieval_score=True
    )
    #nodes = get_nodes_with_scores(result, query=query, top_k=top_k)
    #simlarity_nodes = similarity_cutoff_nodes(result = result, query=query, score=score, top_k=top_k)
    #reranked_nodes = colbert_reranker.postprocess_nodes(nodes, query_str=query)
    reranked_nodes = colbert_reranker.postprocess_nodes(result, query_str=query)
    return reranked_nodes




