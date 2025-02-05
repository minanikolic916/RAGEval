from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval import assert_test

from dotenv import load_dotenv
from statistics import mean
import pandas as pd
import json
from eval_data.eval_utils import create_eval_dataset_deepeval
from log_data.log_utils import read_data

load_dotenv()
FILE_PATH = "./log_data/log_runs/RAG_LOG.json"
EVAL_PATH = "./eval_data/eval_runs/EVAL_LOG_DEEPEVAL.json"
threshold = 0.5
MODEL_NAME = "gpt-4o"
NO_OF_ITERS = 3

# definisaje metrika
faithfulness = FaithfulnessMetric(
    threshold=threshold,
    model= MODEL_NAME,
    include_reason=True, 
)
answer_relevancy = AnswerRelevancyMetric(
    threshold=threshold,
    model= MODEL_NAME,
    include_reason=True
)
contextual_relevancy = ContextualRelevancyMetric(
    threshold=threshold,
    model= MODEL_NAME,
    include_reason=True
)

def calculate_average_metrics_deepeval(file_path:str, iterations:int):
    #dataset.evaluate([faithfulness, answer_relevancy, contextual_relevancy])
    data_list = []
    dataset = create_eval_dataset_deepeval(file_path)
    for _ in range(iterations):
        for test_case in dataset:
            #assert_test(test_case, [faithfulness, answer_relevancy, contextual_relevancy])
            print(test_case)
            faithfulness.measure(test_case)
            answer_relevancy.measure(test_case)
            contextual_relevancy.measure(test_case)
            data_list.append({
                'faithfulness': faithfulness.score, 
                'answer_relevancy':answer_relevancy.score, 
                'contextual_relevancy':contextual_relevancy.score
            })
    df = pd.DataFrame(data_list)
    print(df)
    mean_scores = {key: mean(d[key] for d in data_list) for key in data_list[0]}
    mean_scores.update({"iterations": iterations, "llm_evaluator": MODEL_NAME})
    return mean_scores

def log_eval_data_deepeval(file_path_to_read:str, file_path_to_write:str, iterations:int):
    scores = calculate_average_metrics_deepeval(file_path_to_read, iterations)
    previous_data = read_data(file_path_to_write)
    if not previous_data:
        previous_data = []
    previous_data.append(scores)
    with open(file_path_to_write, 'w') as json_file:
        json.dump(previous_data, json_file, indent=4)
    print(f"Podaci upisani")
    return previous_data

log_eval_data_deepeval(FILE_PATH, EVAL_PATH, NO_OF_ITERS)