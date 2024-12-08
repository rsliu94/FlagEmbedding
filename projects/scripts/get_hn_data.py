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
parser.add_argument("--use_examples_in_query", type=bool, default=True, help="Whether to use the embedder eval data")
parser.add_argument("--model_path", type=str, default="BAAI/bge-en-icl", help="The path of the model")
parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights")
parser.add_argument("--is_submission", type=bool, default=False, help="Whether is submission")
parser.add_argument("--num_examples", type=int, default=1, help="Number of examples per query")
parser.add_argument("--query_max_len", type=int, default=1024, help="The maximum length of the query")
parser.add_argument("--doc_max_len", type=int, default=128, help="The maximum length of the document")
parser.add_argument("--output_path", type=str, default="./test.jsonl", help="The path of the output file")
parser.add_argument("--negative_number", type=int, default=15, help="The number of negative samples per query")
parser.add_argument("--range_for_sampling", type=str, default="2-200", help="The range for sampling")
parser.add_argument("--shuffle_data", type=bool, default=True, help="Whether to shuffle the data")
parser.add_argument("--batch_size", type=int, default=4, help="The batch size")
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
        queries_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "train_queries.jsonl")
        examples_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "examples.json")
    else:
        EMBEDDER_EVAL_DATA_DIR = os.path.join(PROJECT_ROOT, "projects/data/embedder_train_eval_data", "submission")
        corpus_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "corpus.jsonl")
        queries_path = os.path.join(EMBEDDER_EVAL_DATA_DIR, "train_queries.jsonl")
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
    num_examples = args.num_examples
    print(f"Number of examples per query: {num_examples}")
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
                        examples = []
                        N = len(examples_dict[subject_id][construct_id])
                        random_ids = random.sample(list(range(N)), min(N, num_examples))
                        for random_id in random_ids:
                            examples.append(get_detailed_example(examples_dict[subject_id][construct_id][random_id]['instruct'], 
                                                              examples_dict[subject_id][construct_id][random_id]['query'], 
                                                              examples_dict[subject_id][construct_id][random_id]['response']))
                        examples_prefix_list.append('\n\n'.join(examples) + '\n\n')
                    else:
                        random_construct_id = random.choice(list(examples_dict[subject_id].keys()))
                        examples = []
                        N = len(examples_dict[subject_id][random_construct_id])
                        random_ids = random.sample(list(range(N)), min(N, num_examples))
                        for random_id in random_ids:
                            examples.append(get_detailed_example(examples_dict[subject_id][random_construct_id][random_id]['instruct'], 
                                                          examples_dict[subject_id][random_construct_id][random_id]['query'], 
                                                          examples_dict[subject_id][random_construct_id][random_id]['response']))
                        examples_prefix_list.append('\n\n'.join(examples) + '\n\n')
                else:
                    # random sample one
                    random_subject_id = random.choice(list(examples_dict.keys()))
                    random_construct_id = random.choice(list(examples_dict[random_subject_id].keys()))
                    N = len(examples_dict[random_subject_id][random_construct_id])
                    random_ids = random.sample(list(range(N)), min(N, num_examples))
                    for random_id in random_ids:
                        examples.append(get_detailed_example(examples_dict[random_subject_id][random_construct_id][random_id]['instruct'], 
                                                          examples_dict[random_subject_id][random_construct_id][random_id]['query'], 
                                                          examples_dict[random_subject_id][random_construct_id][random_id]['response']))
                    examples_prefix_list.append('\n\n'.join(examples) + '\n\n')
            else:
                examples_prefix_list.append('')
    print(f"Number of examples prefix: {len(examples_prefix_list)}")
    print(f"Number of queries: {len(queries)}")

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
        print(f"Loading LoRA model from {args.lora_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        model = AutoModel.from_pretrained(args.model_path, quantization_config=None)
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)
    else:
        print(f"Loading base model from {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        model = AutoModel.from_pretrained(args.model_path, quantization_config=None)
    
    model = model.half()
    model = model.to(device)
    
    model = model.eval()
    batch_size = args.batch_size
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

    range_start, range_end = map(int, args.range_for_sampling.split('-'))
    print(f"Range for sampling: {range_start} to {range_end}, negative number: {args.negative_number}")
    
    doc_embeddings = inference_doc(corpus, tokenizer, model, args.doc_max_len, batch_size, device)
    query_embeddings = inference_query_examples_list(queries, args.query_max_len, examples_prefix_list, tokenizer, model, batch_size, device)
    print("Building index...")
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    distances, indices = index.search(query_embeddings, k=range_end)
    print(f"Distances shape: {distances.shape}, Indices shape: {indices.shape}")
    
    # write to file
    queries_raw = [json.loads(line) for line in open(queries_path, 'r')] # list of dicts
    new_lines = []
    for line, indice in zip(queries_raw, indices):
        new_line = {}
        new_line['query'] = line['query']
        new_line['pos'] = line['pos']
        new_line['neg'] = []
        sample_pool = indice[range_start:range_end].tolist()
        sampled_ids = random.sample(sample_pool, args.negative_number)
        for id in sampled_ids:
            new_line['neg'].append(corpus[id])
        new_line['prompt'] = line['prompt']
        new_lines.append(new_line)
    
    if args.shuffle_data:
        print("Shuffling data...")
        random.shuffle(new_lines)
    
    print(f"Writing to file: {args.output_path}")
    with open(args.output_path, 'w') as f:
        for line in new_lines:
            f.write(json.dumps(line) + '\n')
    
    print("Done!")
    
    