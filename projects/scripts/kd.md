# Distill reranker to train embedder

## Add reranker score to train data
prompt: 
Predict whether the passage B explains the mathematical misconception that leads to the wrong answer in query A.
detailed instruct:
A: {query} B: {passage} {prompt}

cuda:0 -> 19G VRAM 12min 3506 iters
```bash
python add_reranker_score.py \
--input_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round2.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round2_score.jsonl \
--reranker_name_or_path BAAI/bge-reranker-v2-gemma \
--query_instruction_for_rerank 'A: ' \
--query_instruction_format '{}{}' \
--passage_instruction_for_rerank 'B: ' \
--passage_instruction_format '{}{}' \
--prompt "Predict whether the passage B explains the mathematical misconception that leads to the wrong answer in query A." \
--reranker_max_length 384 \
--use_fp16 True \
--reranker_batch_size 16 \
--devices cuda:0
```

