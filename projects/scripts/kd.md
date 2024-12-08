# Distill reranker to train embedder

<!-- # Iteration 1  [KD in iter 1 is meaningless]

iter0: hn
iter1: hn + ft-ranker
iter2: hn+kd + ft_ranker


## Mine 
* range_for_sampling: [2, 200]
* negative_number: 15
Train data:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/icl_finetune_iter1_hn/merged_model_lora_epoch_2 \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_kd.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:1 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task False \
--batch_size 1024 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```
Test data:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/icl_finetune_iter1_hn/merged_model_lora_epoch_2 \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_test_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_kd_test.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:0 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task False \
--batch_size 1024 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```
## Score
* prompt: 
<!-- Predict whether the passage B explains the mathematical misconception that leads to the wrong answer in query A. -->
Determine whether the passage B contains a mathematical misconception to wrong answer in the query A by providing a prediction of either 'Yes' or 'No'.
* detailed instruct:
A: {query} B: {passage} {prompt}

cuda:0 -> 19G VRAM 12min 3506 iters
cutoff_layers=[28], compress_ratio=2, compress_layers=[24, 40]
Train data:
```bash
python add_reranker_score.py \
--input_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_kd.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_kd.jsonl \
--reranker_name_or_path BAAI/bge-reranker-v2.5-gemma2-lightweight \
--query_instruction_for_rerank 'A: ' \
--query_instruction_format '{}{}' \
--passage_instruction_for_rerank 'B: ' \
--passage_instruction_format '{}{}' \
--prompt "Determine whether the passage B contains a mathematical misconception to wrong answer in the query A by providing a prediction of either 'Yes' or 'No'." \
--reranker_max_length 512 \
--use_fp16 True \
--reranker_batch_size 16 \
--compress_ratio 2 \
--compress_layers 24 40 \
--cutoff_layers 28 \
--devices cuda:0
```
Test data:
```bash
python add_reranker_score.py \
--input_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round2_test.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round2_test_score.jsonl \
--reranker_name_or_path BAAI/bge-reranker-v2.5-gemma2-lightweight \
--query_instruction_for_rerank 'A: ' \
--query_instruction_format '{}{}' \
--passage_instruction_for_rerank 'B: ' \
--passage_instruction_format '{}{}' \
--prompt "Predict whether the passage B explains the mathematical misconception that leads to the wrong answer in query A." \
--reranker_max_length 384 \
--use_fp16 True \
--reranker_batch_size 16 \
--devices cuda:1
```
 -->
