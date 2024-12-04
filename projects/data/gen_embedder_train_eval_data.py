import platform
import os
import subprocess
import argparse
import json
import shutil
import pandas as pd
from FlagEmbedding.utils.data_utils import preprocess_data
from FlagEmbedding.utils.env_utils import get_env_info
from FlagEmbedding.utils.constants import TASK_DESCRIPTION

# Add str2bool helper function
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# Add argument parser
parser = argparse.ArgumentParser(description='Eval data preprocessing script for EEDI Competition')
parser.add_argument('--is_submission', type=str2bool, nargs='?', const=True, default=False,
                   help='Whether is for submission (true/false)')
args = parser.parse_args()

# Add this after parsing arguments
print("\nScript arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print() 

if __name__ == "__main__":
    env_name, PROJECT_ROOT = get_env_info()
    
    print(f"Running on {env_name}, project root: {PROJECT_ROOT}")
    if args.is_submission:
        RAW_DATA_DIR = f"{PROJECT_ROOT}/projects/data/raw_data"
        OUTPUT_DIR = f"{PROJECT_ROOT}/projects/data/embedder_train_eval_data/submission"
    else:
        RAW_DATA_DIR = f"{PROJECT_ROOT}/projects/data/raw_data/cross_validation"
        OUTPUT_DIR = f"{PROJECT_ROOT}/projects/data/embedder_train_eval_data/cross_validation"
    
    # mkdir if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    misconception_mapping = pd.read_csv(f"{RAW_DATA_DIR}/misconception_mapping.csv")
    misconception_mapping['MisconceptionName'] = misconception_mapping['MisconceptionName'].str.strip()
    corpus = misconception_mapping['MisconceptionName'].values.tolist()
    
    # Generate corpus.jsonl
    with open(f"{OUTPUT_DIR}/corpus.jsonl", "w", encoding="utf-8") as f:
        for sentence in corpus:
            json_line = {"text": sentence}  # 将每一行封装为字典
            f.write(json.dumps(json_line) + "\n")
    
    print(f"Reading train data from {RAW_DATA_DIR}/train.csv")
    train_data = pd.read_csv(f"{RAW_DATA_DIR}/train.csv")
    train_preprocessed = preprocess_data(train_data, misconception_mapping, 
                                        with_misconception=True, 
                                        filter_na_misconception=True)
    print(f"Reading test data from {RAW_DATA_DIR}/test.csv")
    test_data = pd.read_csv(f"{RAW_DATA_DIR}/test.csv")
    if args.is_submission:
        test_preprocessed = preprocess_data(test_data, misconception_mapping, 
                                            with_misconception=False, 
                                            filter_na_misconception=True)
    else:
        test_preprocessed = preprocess_data(test_data, misconception_mapping, 
                                            with_misconception=True, 
                                            filter_na_misconception=True)
    
    with open(f"{OUTPUT_DIR}/train_queries.jsonl", "w", encoding="utf-8") as f:
        for _, row in train_preprocessed.iterrows():
            misconception_id = row['MisconceptionId']
            misconception_name = row['MisconceptionName'].strip()
            question_text = row['QuestionText'].strip()
            subject_name = row['SubjectName'].strip()
            construct_name = row['ConstructName'].strip()
            correct_answer = row['CorrectAnswerText'].strip()
            wrong_answer = row['WrongAnswerText'].strip()
            subject_id = row['SubjectId']
            construct_id = row['ConstructId']
            question_id = row['QuestionId']
            question_id_answer = row['QuestionId_Answer']
            # build query
            query = f"""Question: {question_text}\nHint: {construct_name}\nCorrect answer: {correct_answer}\nWrong answer: {wrong_answer}"""
            json_line = {
                "query": query,
                "pos": [misconception_name],
                "neg": [],
                "correct_id": misconception_id,
                "question_id_answer": question_id_answer,
                "subject_id": subject_id,
                "construct_id": construct_id,
                "prompt": TASK_DESCRIPTION,
            }
            f.write(json.dumps(json_line) + "\n")
            
    if args.is_submission:
        with open(f"{OUTPUT_DIR}/test_queries.jsonl", "w", encoding="utf-8") as f:
            for _, row in test_preprocessed.iterrows():
                question_text = row['QuestionText'].strip()
                subject_name = row['SubjectName'].strip()
                construct_name = row['ConstructName'].strip()
                correct_answer = row['CorrectAnswerText'].strip()
                wrong_answer = row['WrongAnswerText'].strip()
                subject_id = row['SubjectId']
                construct_id = row['ConstructId']
                question_id = row['QuestionId']
                question_id_answer = row['QuestionId_Answer']
                # build query
                query = f"""Question: {question_text}\nHint: {construct_name}\nCorrect answer: {correct_answer}\nWrong answer: {wrong_answer}"""
                json_line = {
                    "query": query,
                    "question_id_answer": question_id_answer,
                    "subject_id": subject_id,
                    "construct_id": construct_id,
                    "prompt": TASK_DESCRIPTION,
                }
                f.write(json.dumps(json_line) + "\n")
    else:
        with open(f"{OUTPUT_DIR}/test_queries.jsonl", "w", encoding="utf-8") as f:
            for _, row in test_preprocessed.iterrows():
                misconception_id = row['MisconceptionId']
                assert misconception_id != -1
                misconception_name = row['MisconceptionName'].strip()
                question_text = row['QuestionText'].strip()
                subject_name = row['SubjectName'].strip()
                construct_name = row['ConstructName'].strip()
                correct_answer = row['CorrectAnswerText'].strip()
                wrong_answer = row['WrongAnswerText'].strip()
                subject_id = row['SubjectId']
                construct_id = row['ConstructId']
                question_id = row['QuestionId']
                question_id_answer = row['QuestionId_Answer']
                # build query
                query = f"""Question: {question_text}\nHint: {construct_name}\nCorrect answer: {correct_answer}\nWrong answer: {wrong_answer}"""
                json_line = {
                    "query": query,
                    "pos": [misconception_name],
                    "neg": [],
                    "correct_id": misconception_id,
                    "question_id_answer": question_id_answer,
                    "subject_id": subject_id,
                    "construct_id": construct_id,
                    "prompt": TASK_DESCRIPTION,
                }
                f.write(json.dumps(json_line) + "\n")
                
    # if args.is_submission:
    # build a examples dictionary from raw_data/train.csv, then save it to examples.json
    all_train_data = pd.read_csv(f"{PROJECT_ROOT}/projects/data/raw_data/train.csv")
    all_train_preprocessed = preprocess_data(all_train_data, misconception_mapping, 
                                        with_misconception=True,
                                        filter_na_misconception=True)
    
    from collections import defaultdict
    examples = defaultdict(dict)
    for _, row in all_train_preprocessed.iterrows():
        question_text = row['QuestionText'].strip()
        construct_name = row['ConstructName'].strip()
        correct_answer = row['CorrectAnswerText'].strip()
        wrong_answer = row['WrongAnswerText'].strip()
        query = f"""Question: {question_text}\nHint: {construct_name}\nCorrect answer: {correct_answer}\nWrong answer: {wrong_answer}"""
        response = row['MisconceptionName'].strip()
        subject_id = row['SubjectId']
        construct_id = row['ConstructId']
        if construct_id not in examples[subject_id]:
            examples[subject_id][construct_id] = [
                {
                    "instruct": TASK_DESCRIPTION,
                    "query": query,
                    "response": response
                }
            ]
        else:
            examples[subject_id][construct_id].append({
                "instruct": TASK_DESCRIPTION,
                "query": query,
                "response": response
            })
    with open(f"{OUTPUT_DIR}/examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=4)
        

    # Generate hn mine input data
    with open(f"{OUTPUT_DIR}/train_queries.jsonl", "r", encoding="utf-8") as f:
        train_queries = [json.loads(line) for line in f]
        
        
    with open(f"{OUTPUT_DIR}/hn_mine_input.jsonl", "w", encoding="utf-8") as f:
        for line in train_queries:
            json_line = {
                "query": line['query'],
                "pos": line['pos'],
                "neg": line['neg'],
                "prompt": line['prompt'],
            }
            f.write(json.dumps(json_line) + "\n")