from ragas.dataset_schema import SingleTurnSample, EvaluationDataset as RagasEvaluationDataset
from deepeval.dataset import EvaluationDataset as DeepEvalEvaluationDataset 
from log_data.log_utils import read_data

def create_eval_dataset_ragas(file_path:str):
    data = read_data(file_path)
    single_samples_list = []
    for item in data:
        single_sample = SingleTurnSample(
            user_input = item["query"], 
            retrieved_contexts = item["retrieval_context"], 
            response = item["actual_output"]
        )
        single_samples_list.append(single_sample)

    dataset = RagasEvaluationDataset(samples = single_samples_list)
    return dataset 

def create_eval_dataset_deepeval(file_path:str):
    dataset = DeepEvalEvaluationDataset()
    dataset.add_test_cases_from_json_file(
        file_path = file_path, 
        input_key_name="query",
        actual_output_key_name="actual_output",
        retrieval_context_key_name="retrieval_context",
    )
    return dataset 