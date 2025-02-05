import json
import datetime
import os 

def log_data_to_json(file_path, question, context, answer):
    if not isinstance(context, list):
        context = [context]
    log_entry = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "query": question,
        "retrieval_context": context,
        "actual_output": answer
    }

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
        data.append(log_entry)

        with open(file_path, 'w', encoding = "utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii= False)
    else:
        with open(file_path, 'w', encoding = "utf-8") as file:
            json.dump([log_entry], file, indent=4, ensure_ascii= False)

def read_data(file_path, file_format='json'):
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data
    except Exception as e:
        return []

