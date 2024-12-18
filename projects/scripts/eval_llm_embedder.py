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
from FlagEmbedding.utils.infer_utils import inference_doc, inference_query_examples_list, inference_query, batch_to_device
import argparse
import random
import json

# set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# argparser
parser = argparse.ArgumentParser(description="Evaluate the raw LLM embedder")
parser.add_argument("--use_examples_in_query", type=bool, default=False, help="Whether to use the embedder eval data")
parser.add_argument("--model_path", type=str, default="BAAI/bge-en-icl", help="The path of the model")
parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights")
parser.add_argument("--is_submission", type=bool, default=False, help="Whether is submission")
parser.add_argument("--query_max_len", type=int, default=1024, help="The maximum length of the query")
parser.add_argument("--doc_max_len", type=int, default=128, help="The maximum length of the document")
parser.add_argument("--k", type=int, default=25, help="The number of retrieved documents")
parser.add_argument("--batch_size", type=int, default=4, help="The batch size")
parser.add_argument("--save_retrieval_results", type=bool, default=False, help="Whether to save the retrieval results")
parser.add_argument("--retrieval_results_path", type=str, default='./retrieval_results.jsonl', help="The path to save the retrieval results")
parser.add_argument("--device", type=str, default='cuda:0', help="The device to use")
args = parser.parse_args()
# show args
print("\nScript arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print()

if __name__ == "__main__":
    env_name, PROJECT_ROOT = get_env_info()
    if not args.is_submission:
        EMBEDDER_EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "projects/data/embedder_train_eval_data", "cross_validation")
        corpus_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "corpus.jsonl")
        queries_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "test_queries.jsonl")
        examples_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "examples.json")
    else:
        EMBEDDER_EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "projects/data/embedder_train_eval_data", "submission")
        corpus_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "corpus.jsonl")
        queries_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "test_queries.jsonl")
        examples_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "examples.json")
    

    print(f"DEBUG: examples path: {examples_path}, loading examples dict [subject_id -> construct_id -> [instruct, query, response]]")
    with open(examples_path, 'r') as f:
        examples_dict = json.load(f)
    
    corpus = [json.loads(line)['text'] for line in open(corpus_path, 'r')] # list of strings
    print(f"Number of corpus: {len(corpus)}")
    question_ids = [json.loads(line)['question_id_answer'] for line in open(queries_path, 'r')] # list of floats
    print(f"Number of question ids: {len(question_ids)}")
    if not args.is_submission:
        correct_ids = [json.loads(line)['correct_id'] for line in open(queries_path, 'r')] # list of floats
        print(f"Number of correct ids: {len(correct_ids)}")
    
    queries = []
    examples_prefix_list = []
    with open(queries_path, 'r') as f:
        for line in f:
            row = json.loads(line)
            queries.append(get_detailed_instruct(task_description=row['prompt'], query=row['query']))
            subject_id = str(row['subject_id'])
            construct_id = str(row['construct_id'])
            if args.use_examples_in_query:
                if subject_id in examples_dict:
                    if construct_id in examples_dict[subject_id]:
                        random_id = random.choice(list(range(len(examples_dict[subject_id][construct_id]))))
                        examples = [get_detailed_example(examples_dict[subject_id][construct_id][random_id]['instruct'], 
                                                        examples_dict[subject_id][construct_id][random_id]['query'], 
                                                        examples_dict[subject_id][construct_id][random_id]['response'])]
                        examples_prefix_list.append('\n\n'.join(examples) + '\n\n')
                    else:
                        random_construct_id = random.choice(list(examples_dict[subject_id].keys()))
                        random_id = random.choice(list(range(len(examples_dict[subject_id][random_construct_id]))))
                        examples = [get_detailed_example(examples_dict[subject_id][random_construct_id][random_id]['instruct'], 
                                                        examples_dict[subject_id][random_construct_id][random_id]['query'], 
                                                        examples_dict[subject_id][random_construct_id][random_id]['response'])]
                        examples_prefix_list.append('\n\n'.join(examples) + '\n\n')
                else:
                    # random sample one
                    random_subject_id = random.choice(list(examples_dict.keys()))
                    random_construct_id = random.choice(list(examples_dict[random_subject_id].keys()))
                    random_id = random.choice(list(range(len(examples_dict[random_subject_id][random_construct_id]))))
                    examples = [get_detailed_example(examples_dict[random_subject_id][random_construct_id][random_id]['instruct'], 
                                                    examples_dict[random_subject_id][random_construct_id][random_id]['query'], 
                                                    examples_dict[random_subject_id][random_construct_id][random_id]['response'])]
                    examples_prefix_list.append('\n\n'.join(examples) + '\n\n')
            else:
                examples_prefix_list.append('')
    print(f"Number of examples prefix: {len(examples_prefix_list)}")
    print(f"Number of queries: {len(queries)}")

    print("Loading tokenizer and model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.float16
    )
    if args.lora_path is not None:
        print("Loading LoRA tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, 
                                      device_map=args.device,
                                      quantization_config=bnb_config)
    if args.lora_path is not None:
        print("Loading LoRA model from {}".format(args.lora_path))
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)
    
    model = model.eval()
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


    doc_embeddings = inference_doc(corpus, tokenizer, model, args.doc_max_len, args.batch_size, args.device)
    query_embeddings = inference_query_examples_list(queries, args.query_max_len, examples_prefix_list, tokenizer, model, args.batch_size, args.device)
    print("Building index...")
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    distances, indices = index.search(query_embeddings, k=100)
    print(f"Distances shape: {distances.shape}, Indices shape: {indices.shape}")
    
    if not args.is_submission:
        for k in [25, 50, 75, 100]:
            mapk_score = mean_average_precision_at_k(correct_ids, indices, k)
            print(f"map@{k}_score: {mapk_score}")
            # mapk_score_2 = mapk([[id] for id in correct_ids], indices, k)
            # print(f"map@{k}_score_2: {mapk_score_2}")
            recall_score = recall_at_k(correct_ids, indices, k)
            print(f"recall@{k}_score: {recall_score}")
    
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
        
    if args.save_retrieval_results:
        print(f"Saving Top-{args.k} retrieval results... Reading test queries from {queries_path}")
        # query is the same as test_queries.jsonl
        queries = [json.loads(line) for line in open(queries_path, 'r')]
        
        print(f"Number of queries read: {len(queries)}")
        assert len(queries) == query_embeddings.shape[0]
        
        print(f"Retrieving Top-{args.k} results...")
        distances, indices = index.search(query_embeddings, k=args.k)
        
        print(f"Distances shape: {distances.shape}, Indices shape: {indices.shape}")
        corpus = [json.loads(line)['text'] for line in open(corpus_path, 'r')] # list of strings
        
        print(f"Saving retrieval results to {args.retrieval_results_path}")
        with open(args.retrieval_results_path, 'w') as f:
            for query, indice in zip(queries, indices):
                query['candidate_ids'] = indice.tolist()
                query['candidate_texts'] = [corpus[i] for i in indice]
                f.write(json.dumps(query) + '\n')
        print(f"Done!")