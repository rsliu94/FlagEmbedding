import platform
import os
import subprocess
import json
import argparse
import pandas as pd
from FlagEmbedding.utils.env_utils import get_env_info
from FlagEmbedding.utils.data_utils import preprocess_text, preprocess_data

parser = argparse.ArgumentParser(description='Create Input File for Hard Negative Mining')
parser.add_argument('--mode', type=str, choices=['validation', 'submission'],
                   default='validation', help='Dataset mode')
parser.add_argument('--validation_version', type=str, default='v2', help='Validation version')
parser.add_argument('--sub_dir_name', type=str, default='hn_mine_data_round_1', help='Sub directory name')
args = parser.parse_args()

# Add this after parsing arguments
print("\nScript arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print() 
    
def main():
    # Get environment and device setup
    env_name, PROJECT_ROOT = get_env_info()
    
    if args.mode == "submission":
        OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'projects', 'data', args.sub_dir_name, "submission")
    else:
        OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'projects', 'data', args.sub_dir_name, f"validation_{args.validation_version}")
    # create output root if not exists
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    print(f"Running on {env_name}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    
    
    if args.mode == 'validation':
        path = os.path.join(PROJECT_ROOT, 'projects', 'data', 'raw_data', f'validation_{args.validation_version}', 'train.csv')
        print(f"Read from: {path}")
        train_data = pd.read_csv(path)
    elif args.mode == 'submission':
        path = os.path.join(PROJECT_ROOT, 'projects', 'data', 'raw_data', 'train.csv')
        print(f"Read from: {path}")
        train_data = pd.read_csv(path)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    print(f"len(train_data): {len(train_data)}")
    
    misconception_mapping = pd.read_csv(os.path.join(PROJECT_ROOT, 'projects', 'data', 'raw_data', 'misconception_mapping.csv'))
    # candidate pool
    candidate_pool = [{'text': misconception} for misconception in list(misconception_mapping['MisconceptionName'].values)]
    
    # preprocess train data
    print("Preprocessing train data...")
    train_data_preprocessed = preprocess_data(train_data, misconception_mapping, 
                                               query_text_version='v1',
                                               with_instruction=False, 
                                               with_misconception=True, 
                                               filter_na_misconception=True)
    train_data_preprocessed['query'] = train_data_preprocessed.apply(
        lambda row: f"###Skill: {row['ConstructName']} ###Question: {row['QuestionText']} ###Incorrect Answer: {row['WrongAnswerText']}", axis=1)
    
    finetune_data = [
        {
            'query': query,
            'pos': [misconception_name],
            'neg': [],
            'prompt': f'Given a math question and a misconcepted incorrect answer to it, retrieve the most accurate misconception that leads to the incorrect answer.'
        } for query, misconception_name in train_data_preprocessed[['query', 'MisconceptionName']].values
    ]
    print(f"len(candidate_pool): {len(candidate_pool)}, len(finetune_data): {len(finetune_data)}")
    
    with open(os.path.join(OUTPUT_ROOT, f'candidate_pool.jsonl'), 'w') as f:
        for entry in candidate_pool:
            json.dump(entry, f)
            f.write('\n')
    print(f"candidate_pool saved to {os.path.join(OUTPUT_ROOT, f'candidate_pool.jsonl')}")
        
    with open(os.path.join(OUTPUT_ROOT, f'finetune_data.jsonl'), 'w') as f:
        for entry in finetune_data:
            json.dump(entry, f)
            f.write('\n')
    print(f"finetune_data saved to {os.path.join(OUTPUT_ROOT, f'finetune_data.jsonl')}")
    
    
if __name__ == "__main__":
    main()
    
    