import platform
import os
import subprocess
import argparse
import json

import pandas as pd
from FlagEmbedding.utils.data_utils import preprocess_data
from FlagEmbedding.utils.env_utils import get_env_info

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
parser.add_argument('--filter_na_misconception', type=str2bool, nargs='?', const=True, default=False,
                   help='Whether to filter out rows with NA in MisconceptionId (true/false)')
parser.add_argument('--with_instruction', type=str2bool, nargs='?', const=True, default=False,
                   help='Whether to add instruction to the query (true/false)')
parser.add_argument('--query_text_version', type=str, choices=['v1', 'v2', 'v3'], default='v1',
                   help='Query text version')
args = parser.parse_args()

# Add this after parsing arguments
print("\nScript arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print() 

if __name__ == "__main__":
    env_name, PROJECT_ROOT = get_env_info()
    RAW_DATA_DIR = f"{PROJECT_ROOT}/projects/data/raw_data"
    OUTPUT_DIR = f"{PROJECT_ROOT}/projects/data/embedder_eval_data"
    # mkdir if not exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    misconception_mapping = pd.read_csv(f"{RAW_DATA_DIR}/misconception_mapping.csv")
    misconception_mapping['MisconceptionName'] = misconception_mapping['MisconceptionName']
    corpus = misconception_mapping['MisconceptionName'].values.tolist()
        
    
    with open(f"{OUTPUT_DIR}/corpus.jsonl", "w", encoding="utf-8") as f:
        for sentence in corpus:
            json_line = {"text": sentence}  # 将每一行封装为字典
            f.write(json.dumps(json_line) + "\n")
    
    for version in ["v1", "v2"]:
        val_data = pd.read_csv(f"{RAW_DATA_DIR}/validation_{version}/val.csv")
        val_preprocessed = preprocess_data(val_data, misconception_mapping, 
                                    query_text_version=args.query_text_version,
                                    with_instruction=args.with_instruction, 
                                    with_misconception=True, 
                                    filter_na_misconception=args.filter_na_misconception)
        
        selected_columns = ["query_text", "QuestionText", "MisconceptionId", "SubjectName", "ConstructName", "CorrectAnswerText", "WrongAnswerText"]
        df_selected = val_preprocessed[selected_columns]

        # 将 JSON 数据写入文件
        with open(f"{OUTPUT_DIR}/queries_{version}.jsonl", "w") as f:
            for _, row in df_selected.iterrows():
                json_line = {"text": row['query_text'], "correct_id": row['MisconceptionId'], 'question': row['QuestionText'], 'subject_name': row['SubjectName'], 'construct_name': row['ConstructName'], 'correct_answer': row['CorrectAnswerText'], 'wrong_answer': row['WrongAnswerText']}
                f.write(json.dumps(json_line) + "\n")