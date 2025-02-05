from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from ragas import evaluate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
from eval_data.eval_utils import create_eval_dataset_ragas
from log_data.log_utils import read_data
from datasets import Dataset 

load_dotenv()
FILE_PATH = "./log_data/log_runs/RAG_LOG.json"
EVAL_PATH = "./eval_data/eval_runs/EVAL_LOG_RAGAS.json"
NO_OF_ITERS = 7
MODEL_NAME = "gpt-4o-mini"

# model za evaluaciju 
evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model= MODEL_NAME))

# definisanje metrika
faithfulness = Faithfulness(llm=evaluator_llm)
context_precision = LLMContextPrecisionWithoutReference(llm = evaluator_llm)
response_relevancy = ResponseRelevancy(llm = evaluator_llm)
metrics = [faithfulness, context_precision, response_relevancy]

def run_evaluation(file_path:str, iterations:int):
    dataset = create_eval_dataset_ragas(file_path)
    results = []
    for _ in range(iterations):
        result = evaluate(dataset, metrics)
        results.append(result)
    return results

def calculate_average_metrics_ragas(file_path:str, iterations:int):
    results = run_evaluation(file_path, iterations)    
    # results = [{'faithfulness': 1.0000, 'llm_context_precision_without_reference': 1.0000, 'answer_relevancy': 0.9464}, 
    # {'faithfulness': 1.0000, 'llm_context_precision_without_reference': 1.0000, 'answer_relevancy': 0.9287}]
    print("-----------------Results---------------------------")
    print(results)
    scores_dict = [score.scores for score in results]
    print("-----------------Scores dict-----------------------")
    print(scores_dict)
    flat_scores = [item for sublist in scores_dict for item in sublist]
    print("-----------------Flat scores-----------------------")
    print(flat_scores)
    cleaned_scores = [{key: float(value) if isinstance(value, np.float64) else value for key, value in entry.items()} for entry in flat_scores]
    mean_scores = {key: sum(d[key] for d in cleaned_scores) / len(cleaned_scores) for key in cleaned_scores[0]}
    print("-----------------Mean values-----------------------")
    print(mean_scores)
    #azuriranje za jos metrika
    mean_scores.update({"iterations": iterations, "llm_evaluator": MODEL_NAME})
    return mean_scores

def log_eval_data_ragas(file_path_to_read:str, file_path_to_write:str, iterations:int):
    scores = calculate_average_metrics_ragas(file_path_to_read, iterations)
    previous_data = read_data(file_path_to_write)
    if not previous_data:
        previous_data = []
    previous_data.append(scores)
    with open(file_path_to_write, 'w') as json_file:
        json.dump(previous_data, json_file, indent=4)
    print(f"Podaci upisani")
    return previous_data

log_eval_data_ragas(FILE_PATH, EVAL_PATH, NO_OF_ITERS)

