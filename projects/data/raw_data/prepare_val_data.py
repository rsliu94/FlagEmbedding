import platform
import os
import subprocess
import numpy as np
import pandas as pd
from FlagEmbedding.utils import get_env_info

def prepare_val_data(env_name, split_method, output_dir, PROJECT_ROOT=None):
    np.random.seed(42)
    if env_name == "Kaggle":
        path = "/kaggle/input/eedi-mining-misconceptions-in-mathematics/train.csv"  
        print(f"Running on Kaggle from {path}")
        raw_data = pd.read_csv(path)
    else:
        path = f"{PROJECT_ROOT}/projects/data/raw_data/train.csv"   
        print(f"Running on {env_name} from {path}")
        raw_data = pd.read_csv(path)
    print(f"Original raw_data shape: {raw_data.shape}")
    
    if split_method == "v2":
        # randomly choose 33.3% QuestionId from train_data as validation set (all the rows contains no NaN MisconceptionId), and the rest as training set
        qualified_question_ids = []
        for id, row in raw_data.iterrows():
            correct_answer = row['CorrectAnswer']
            wrong_answers = [f'Misconception{answer}Id' for answer in ['A', 'B', 'C', 'D'] if answer != correct_answer]
            if all(row[wrong_answers].notna()):
                qualified_question_ids.append(row['QuestionId'])
        print(f"qualified_question_ids shape: {len(qualified_question_ids)}")
        qualified_question_ids = sorted(qualified_question_ids)
        val_question_ids = np.random.choice(qualified_question_ids, int(len(raw_data) * 0.333), replace=False)
    elif split_method == "v1":
        # randomly choose 33.3% QuestionId from train_data as validation set, and the rest as training set
        # Sort QuestionIds to ensure consistent ordering before selection
        question_ids = sorted(raw_data["QuestionId"].unique())
        val_question_ids = np.random.choice(question_ids, int(len(raw_data) * 0.333), replace=False)
    else:
        raise ValueError(f"Invalid split method: {split_method}")
    
    val_data = raw_data[raw_data["QuestionId"].isin(val_question_ids)]
    train_data = raw_data[~raw_data["QuestionId"].isin(val_question_ids)]
    print(f"val_data shape: {val_data.shape}")
    print(f"train_data shape: {train_data.shape}")
    
    print(f"Saving to {output_dir}...")
    # create output_dir if not exists
    os.makedirs(output_dir, exist_ok=True)
    val_data.to_csv(f"{output_dir}/test.csv", index=False)
    print(f"Saved to {output_dir}/test.csv")
    train_data.to_csv(f"{output_dir}/train.csv", index=False)
    print(f"Saved to {output_dir}/train.csv")
    return


if __name__ == "__main__":
    env_name, PROJECT_ROOT = get_env_info()
    print(f"Running on {env_name}")
    print(f"Project root: {PROJECT_ROOT}")
    output_dir = f"{PROJECT_ROOT}/projects/data/raw_data"

    prepare_val_data(env_name, "v1", os.path.join(output_dir, "validation_v1"), PROJECT_ROOT) 
    prepare_val_data(env_name, "v2", os.path.join(output_dir, "validation_v2"), PROJECT_ROOT)
