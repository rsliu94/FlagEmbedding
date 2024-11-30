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
args = parser.parse_args()

# Add this after parsing arguments
print("\nScript arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print() 
    
def main():
    # Get environment and device setup
    env, PROJECT_ROOT = get_env_info()
    OUTPUT_ROOT = os.path.join(PROJECT_ROOT, 'projects', 'data', 'hn_mine_data_zero_round')
    # create output root if not exists
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)
    print(f"Running on {env}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    
    
    if args.mode == 'validation':
        train_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'projects', 'data', 'raw_data', 'validation_v2', 'train.csv'))
    elif args.mode == 'submission':
        train_data = pd.read_csv(os.path.join(PROJECT_ROOT, 'projects', 'data', 'raw_data', 'train.csv'))
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
    
    with open(os.path.join(OUTPUT_ROOT, f'candidate_pool_{args.mode}.jsonl'), 'w') as f:
        for entry in candidate_pool:
            json.dump(entry, f)
            f.write('\n')
    print(f"candidate_pool saved to {os.path.join(OUTPUT_ROOT, f'candidate_pool_{args.mode}.jsonl')}")
        
    with open(os.path.join(OUTPUT_ROOT, f'finetune_data_{args.mode}.jsonl'), 'w') as f:
        for entry in finetune_data:
            json.dump(entry, f)
            f.write('\n')
    print(f"finetune_data saved to {os.path.join(OUTPUT_ROOT, f'finetune_data_{args.mode}.jsonl')}")
    
    
if __name__ == "__main__":
    main()
    
    