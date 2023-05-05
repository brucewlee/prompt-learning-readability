import textstat
from tqdm import tqdm

from utils import file_handler as f_handler

def tester(file_path: str):
    total = 0
    correct = 0
    df = f_handler.load_ndjson_and_return_pandas(file_path)
    lines = df.to_dict('records')
    for line_dict in tqdm(lines):
        text_score = textstat.flesch_kincaid_grade(line_dict["text"])
        text_pair_score = textstat.flesch_kincaid_grade(line_dict["text_pair"])
        true_label = line_dict["difficult_text"]
        if text_score > text_pair_score:
            pred_label = 'text'
        else:
            pred_label = 'text_pair'
        if true_label == pred_label:
            correct += 1
        total += 1
    return correct/total
               

print(tester("datasets/final_OSEN_test.json"))
print(tester("datasets/final_NEWS_test.json"))
print(tester("datasets/final_CCSB.json"))
print(tester("datasets/final_CAMB.json"))
