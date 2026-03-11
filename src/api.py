import os
from openai import OpenAI
from run_code import PROJECT_ROOT, BASE_URL, API_KEY, model

os.makedirs(os.path.join(PROJECT_ROOT, f"batch_inputs/{model}/"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, f"predictions/{model}/"), exist_ok=True)
requests_file = os.path.join(PROJECT_ROOT, f"batch_inputs/{model}/all_test.jsonl")
results_file = os.path.join(PROJECT_ROOT, f"predictions/{model}/all_test.jsonl")
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

if __name__ == "__main__":
    batch_requests = client.files.create(file=open(requests_file, "rb"), purpose="batch")
    batch_1 = client.batches.create(input_file_id=batch_requests.id, endpoint="/v1/chat/completions", completion_window="24h", metadata={"description": "Asynchronous job"})
    batch_result = client.files.content(batch_1.output_file_id)
    file = open(results_file, "w+")
    file.write(batch_result.text)
    client.files.delete(batch_1.output_file_id)

