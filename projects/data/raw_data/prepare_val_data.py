import platform
import os
import subprocess
import numpy as np
import pandas as pd
from FlagEmbedding.utils import get_env_info

def prepare_val_data(env_name, output_dir, PROJECT_ROOT=None):
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
    
    question_ids = sorted(raw_data["QuestionId"].unique())
    val_question_ids = np.random.choice(question_ids, int(len(raw_data) * 0.2), replace=False)
    
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

    prepare_val_data(env_name, os.path.join(output_dir, "cross_validation"), PROJECT_ROOT)