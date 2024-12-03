import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
import numpy as np
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import faiss
import os
import pandas as pd
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from FlagEmbedding.utils.metrics import mapk, apk, mean_average_precision_at_k, recall_at_k
from FlagEmbedding.utils.data_utils import preprocess_text, preprocess_data
from FlagEmbedding.utils.env_utils import get_env_info
from FlagEmbedding.utils.format_utils import get_detailed_example, get_detailed_instruct
from FlagEmbedding.utils.infer_utils import inference_doc, inference_query, batch_to_device
import argparse
import json

# argparser
parser = argparse.ArgumentParser(description="Evaluate the raw LLM embedder")
parser.add_argument("--use_examples_in_query", type=bool, default=False, help="Whether to use the embedder eval data")
parser.add_argument("--validation_version", type=str, default="2", help="The version of the validation data")
parser.add_argument("--model_path", type=str, default="BAAI/bge-en-icl", help="The path of the model")
parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights")
parser.add_argument("--is_submission", type=bool, default=False, help="Whether is submission")
args = parser.parse_args()
# show args
print("\nScript arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print()

if __name__ == "__main__":
    env_name, PROJECT_ROOT = get_env_info()
    if not args.is_submission:
        EMBEDDER_EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "projects/data/embedder_eval_data", f"validation_v{args.validation_version}")
        corpus_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "corpus.jsonl")
        queries_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "queries.jsonl")
        examples_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "examples.json")
    else:
        EMBEDDER_EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "projects/data/embedder_eval_data", f"submission")
        corpus_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "corpus.jsonl")
        queries_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "queries.jsonl")
        examples_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "examples.json")
    

    task = 'Given a math question and a misconcepted incorrect answer to it, retrieve the most accurate reason for the misconception leading to the incorrect answer.'
    with open(examples_path, 'r', encoding='utf-8') as f:
        examples = json.load(f)
    
    examples = [get_detailed_example(e['instruct'], e['query'], e['response']) for e in examples]
    
    if args.use_examples_in_query:
        examples_prefix = '\n\n'.join(examples) + '\n\n' # if there not exists any examples, just set examples_prefix = ''
    else:
        examples_prefix = ''
    
    corpus = [json.loads(line)['text'] for line in open(corpus_path, 'r')] # list of strings
    print(f"Number of corpus: {len(corpus)}")
    question_ids = [json.loads(line)['QuestionId_Answer'] for line in open(queries_path, 'r')] # list of floats
    print(f"Number of question ids: {len(question_ids)}")
    if not args.is_submission:
        correct_ids = [json.loads(line)['correct_id'] for line in open(queries_path, 'r')] # list of floats
        print(f"Number of correct ids: {len(correct_ids)}")
    
    queries = []
    with open(queries_path, 'r') as f:
        for line in f:
            row = json.loads(line)
            task_description = f'Given a math question about {preprocess_text(row["construct_name"])} and a misconcepted incorrect answer to it, retrieve the most accurate reason for the misconception leading to the incorrect answer.'
            query = f'{row["question"]} \n Incorrect answer : {row["wrong_answer"]}'
            queries.append(get_detailed_instruct(task_description=task_description, query=query))
    print(f"Number of queries: {len(queries)}")

    query_max_len, doc_max_len = 384, 128

    print("Loading tokenizer and model...")
    # bnb_config = BitsAndBytesConfig(
    #             load_in_4bit=False,
    #             load_in_8bit=False,
    #             bnb_4bit_compute_dtype=torch.float16,
    #             llm_int8_has_fp16_weight=True,
    #         )
    
    # bnb_config = BitsAndBytesConfig(
    #             load_in_4bit=True,
    #             bnb_4bit_use_double_quant=True,
    #             bnb_4bit_quant_type="nf4",
    #             bnb_4bit_compute_dtype=torch.bfloat16
    #         )
    
    if args.lora_path is not None:
        print("Loading LoRA model...")
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        model = AutoModel.from_pretrained(args.model_path, quantization_config=None)
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)
    else:
        print("Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        model = AutoModel.from_pretrained(args.model_path, quantization_config=None)
    
    model = model.half()
    model = model.to(device)
    
    model = model.eval()
    batch_size = 4
    print(f"Model device: {next(model.parameters()).device}")

    print("Check query/document token length...")
    cur_query_max_len = 0
    for query in tqdm(queries):
        cur_query_max_len = max(cur_query_max_len, len(tokenizer(query)['input_ids']))
    print(f"Current query max length: {cur_query_max_len}")

    cur_doc_max_len = 0
    for doc in tqdm(corpus):
        cur_doc_max_len = max(cur_doc_max_len, len(tokenizer(doc)['input_ids']))
    print(f"Current document max length: {cur_doc_max_len}")


    doc_embeddings = inference_doc(corpus, tokenizer, model, doc_max_len, batch_size, device)
    query_embeddings = inference_query(queries, query_max_len, examples_prefix, tokenizer, model, batch_size, device)
    print("Building index...")
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    distances, indices = index.search(query_embeddings, k=25)
    print(f"Distances shape: {distances.shape}, Indices shape: {indices.shape}")
    
    if not args.is_submission:
        mapk_score = mean_average_precision_at_k(correct_ids, indices, 25)
        print(f"map@25_score: {mapk_score}")
        
        mapk_score_2 = mapk([[id] for id in correct_ids], indices, 25)
        print(f"map@25_score_2: {mapk_score_2}")

        recall_score = recall_at_k(correct_ids, indices, 25)
        print(f"recall@25_score: {recall_score}")
    
    if not args.is_submission:
        df = pd.DataFrame({
            'QuestionId_Answer': question_ids,
            'CorrectId': correct_ids,
            'MisconceptionId': [' '.join(map(str, c)) for c in indices.tolist()]
        })
        df.to_csv("./submission.csv", index=False)
    else:
        df = pd.DataFrame({
            'QuestionId_Answer': question_ids,
            'MisconceptionId': [' '.join(map(str, c)) for c in indices.tolist()]
        })
        df.to_csv("./submission.csv", index=False)
        
    import pickle
    np.save('query_embeddings.npy', query_embeddings)
    np.save('doc_embeddings.npy', doc_embeddings)
    with open('queries.pkl', 'wb') as f:
        pickle.dump(queries, f)
    with open('corpus.pkl', 'wb') as f:
        pickle.dump(corpus, f)

    with open('queries.pkl', 'rb') as f:
        loaded_queries = pickle.load(f)
    print(loaded_queries == queries)
    query_embeddings_load = np.load('query_embeddings.npy')
    print(np.array_equal(query_embeddings_load, query_embeddings))