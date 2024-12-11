import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import Tensor
import numpy as np
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, AutoModelForCausalLM
import torch
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
from FlagEmbedding.utils.infer_utils import inference_doc, inference_query_examples_list, inference_query, batch_to_device, get_inputs
from FlagEmbedding.utils.constants import RERANKER_PROMPT
import argparse
import random
import json
from safetensors.torch import load_file
from threading import Lock, Thread
# set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# argparser
parser = argparse.ArgumentParser(description="Evaluate the raw LLM reranker")
parser.add_argument("--retrieval_results_path", type=str, default='./retrieval_results.jsonl', help="The path to save the retrieval results")
parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-14B-Instruct", help="The path of the model")
parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights")
parser.add_argument("--is_submission", type=bool, default=False, help="Whether is submission")
parser.add_argument("--query_max_len", type=int, default=512, help="The maximum length")
parser.add_argument("--doc_max_len", type=int, default=128, help="The maximum length")
parser.add_argument("--k", type=int, default=25, help="The number of retrieved documents")
parser.add_argument("--save_reranker_results", type=bool, default=False, help="Whether to save the reranker results")
parser.add_argument("--devices", type=str, default='cuda:0,cuda:1', help="The devices to use, separated by commas")
parser.add_argument("--batch_size", type=int, default=8, help="The batch size")
parser.add_argument("--device", type=str, default="cuda:0", help="The device")
args = parser.parse_args()
# show args
print("\nScript arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print()

if __name__ == "__main__":
    devices = args.devices.split(',')
    print(f"DEBUG: devices: {devices}")
    models = []
    
    for device in devices:
        print(f"Loading tokenizer and model... on {device}")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.float16
        )
        if args.lora_path is not None:
            print("Loading LoRA tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        model = AutoModelForCausalLM.from_pretrained(args.model_path, 
                                        device_map=device,
                                        quantization_config=bnb_config)
        if args.lora_path is not None:
            print("Loading LoRA model from {}".format(args.lora_path))
            model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)
        model = model.eval()
        models.append((model, tokenizer))
    
    if args.lora_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
    
    retrievals = [json.loads(line) for line in open(args.retrieval_results_path, 'r')] # containing top-25 candidates retrieved by embedder, produced by eval_llm_embedder.py
    pairs = []
    for retrieval in retrievals:
        query = retrieval['query']
        candidates = retrieval['candidate_texts']
        pairs.extend( [[query, candidate] for candidate in candidates])
        
    n_gpu = len(devices)
    assert n_gpu == 2
    
    # pairs_parts = np.array_split(pairs, n_gpu)
    len_pairs = [len(tokenizer(pair[0])['input_ids']) + len(tokenizer(pair[1])['input_ids']) for pair in pairs]
    sorted_indices = sorted(range(len(len_pairs)), key=lambda k: len_pairs[k], reverse=True)
    pairs_part_1_idx = [sorted_indices[i] for i in range(len(sorted_indices)) if i % 2 == 0]
    pairs_part_2_idx = [sorted_indices[i] for i in range(len(sorted_indices)) if i % 2 == 1]
    pairs_part_1 = [pairs[i] for i in pairs_part_1_idx]
    pairs_part_2 = [pairs[i] for i in pairs_part_2_idx]
    pairs_parts = [pairs_part_1, pairs_part_2]
    
    score_results = [None] * n_gpu
    score_results_lock = Lock()
    
    def infer_pairs(pairs, tokenizer, model, yes_loc, device, query_max_len, doc_max_len, batch_size=32):
        scores = []
        for i in tqdm(range(0, len(pairs), batch_size), desc="Evaluating Metrics"):
            batch_pairs = pairs[i:i+batch_size]
            batch_inputs = get_inputs(batch_pairs, prompt=RERANKER_PROMPT, tokenizer=tokenizer, query_max_len=query_max_len, doc_max_len=doc_max_len)
            batch_inputs = batch_to_device(batch_inputs, device)
            scores_tensor = model(**batch_inputs, return_dict=True).logits[:, -1, yes_loc].view(-1, ).float()
            scores.extend(scores_tensor.tolist())
        return scores
    
    def run_inference_pairs(model, tokenizer, pairs_part, device, index):
        result = infer_pairs(pairs_part, tokenizer, model, yes_loc, device, args.query_max_len, args.doc_max_len, args.batch_size)
        with score_results_lock:
            score_results[index] = result
        
    threads = []
    for index, device in enumerate(devices):
        thread = Thread(target=run_inference_pairs, args=(models[index][0], models[index][1], pairs_parts[index], device, index))
        threads.append(thread)
        
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
        
    print(f"DEBUG: len score_results: {len(score_results)}")
    scores_unsorted = score_results[0] + score_results[1]
    print(f"DEBUG: scores_unsorted.shape: {len(scores_unsorted)}")
    original_indexes = pairs_part_1_idx + pairs_part_2_idx
    # Pair each score with its original index
    paired_list = list(zip(scores_unsorted, original_indexes))

    # Sort the paired list based on the original indexes
    sorted_paired_list = sorted(paired_list, key=lambda x: x[1])

    # Extract the scores in original order
    scores = [score for score, index in sorted_paired_list]
    print(f"DEBUG: scores length: {len(scores)}")
    
    # 处理每个检索结果的重排序
    score_idx = 0
    for retrieval in retrievals:
        num_candidates = len(retrieval['candidate_texts'])
        # 获取当前查询的所有候选文档得分
        current_scores = scores[score_idx:score_idx + num_candidates]
        # 将候选ID和得分配对并排序
        id_score_pairs = list(zip(retrieval['candidate_ids'], current_scores))
        sorted_pairs = sorted(id_score_pairs, key=lambda x: x[1], reverse=True)
        # 提取排序后的ID
        retrieval['reranked_ids'] = [pair[0] for pair in sorted_pairs]
        score_idx += num_candidates
    
    # 准备评估数据
    correct_ids = [retrieval['correct_id'] for retrieval in retrievals]
    reranked_ids = [retrieval['reranked_ids'] for retrieval in retrievals]
    recall_ids = [retrieval['candidate_ids'] for retrieval in retrievals]

    # 计算评估指标
    print("==Rerank==")
    mapk_score = mean_average_precision_at_k(correct_ids, np.array(reranked_ids), 25)
    print(f"map@25_score: {mapk_score}")

    recall_score = recall_at_k(correct_ids, np.array(reranked_ids), 25)
    print(f"recall@25_score: {recall_score}")

    print("==Recall==")
    mapk_score = mean_average_precision_at_k(correct_ids, np.array(recall_ids), 25)
    print(f"map@25_score: {mapk_score}")

    recall_score = recall_at_k(correct_ids, np.array(recall_ids), 25)
    print(f"recall@25_score: {recall_score}")