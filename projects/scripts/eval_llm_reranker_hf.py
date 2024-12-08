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

# set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# argparser
parser = argparse.ArgumentParser(description="Evaluate the raw LLM reranker")
parser.add_argument("--retrieval_results_path", type=str, default='./retrieval_results.jsonl', help="The path to save the retrieval results")
parser.add_argument("--model_path", type=str, default="BAAI/bge-reranker-v2-gemma", help="The path of the model")
parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights")
parser.add_argument("--is_submission", type=bool, default=False, help="Whether is submission")
parser.add_argument("--max_len", type=int, default=512, help="The maximum length")
parser.add_argument("--k", type=int, default=100, help="The number of retrieved documents")
parser.add_argument("--save_reranker_results", type=bool, default=False, help="Whether to save the reranker results")
parser.add_argument("--batch_size", type=int, default=8, help="The batch size")
parser.add_argument("--device", type=str, default="cuda:0", help="The device")
args = parser.parse_args()
# show args
print("\nScript arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print()

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16)
    yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
    model.eval()
    model = model.to(args.device)
    
    retrievals = [json.loads(line) for line in open(args.retrieval_results_path, 'r')]
    pairs = []
    for retrieval in retrievals:
        query = retrieval['query']
        candidates = retrieval['candidate_texts']
        pairs.extend( [[query, candidate] for candidate in candidates])
        
    scores = []
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), args.batch_size), desc="Evaluating Metrics"):
            batch_pairs = pairs[i:i+args.batch_size]
            batch_inputs = get_inputs(batch_pairs, prompt=RERANKER_PROMPT, tokenizer=tokenizer, max_length=512)
            batch_inputs = batch_to_device(batch_inputs, next(model.parameters()).device)
            scores_tensor = model(**batch_inputs, return_dict=True).logits[:, -1, yes_loc].view(-1, ).float()
            scores.extend(scores_tensor.tolist())
    print(f"DEBUG: scores length: {len(scores)}")
    # scores = model.compute_score(pairs)
    
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
    
    
    
# # usage
# ```bash
# python eval_llm_reranker.py \
# --retrieval_results_path ../model_output/icl_finetune_iter1_hn/retrieval_results.jsonl \
# --model_path BAAI/bge-reranker-v2.5-gemma2-lightweight \
# --batch_size 8 \
# --device cuda:2
# ```


# num_to_score = 864*100, bs=16, peak=19G
# cutoff_layers=[25], compress_ratio=2, compress_layers=[8]
# 1hr infer

# cutoff_layers=[28], compress_ratio=2, compress_layers=[24, 40]
# 1hr05min infer, bs=8, peak=18G
# ==Rerank==
# map@25_score: 0.23624466721795276
# recall@25_score: 0.7442129629629629
# ==Recall==
# map@25_score: 0.4970577497482009
# recall@25_score: 0.8819444444444444

# KD from pretrained ranker is meaningless. We need to finetune ranker first!


# python eval_llm_reranker.py \
# --retrieval_results_path ../model_output/icl_finetune_iter1_hn/retrieval_results.jsonl \
# --model_path BAAI/bge-reranker-v2-minicpm-layerwise \
# --batch_size 8 \
# --device cuda:2

# minicpm : speed! 25min + low resources: 6G
# ==Rerank==
# map@25_score: 0.20568411986350413
# recall@25_score: 0.7025462962962963
# ==Recall==
# map@25_score: 0.4970577497482009
# recall@25_score: 0.8819444444444444