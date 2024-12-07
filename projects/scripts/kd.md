# Distill reranker to train embedder

## Mine HN using finetuned embedder round 2
* range_for_sampling: [2, 100]
* negative_number: 15

Train data:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/icl_hn_finetune_round2/merged_model_lora_epoch_2 \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round2.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-100 \
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
--embedder_name_or_path ../model_output/icl_hn_finetune_round2/merged_model_lora_epoch_2 \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_test_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round2_test.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-100 \
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

## Add pretrained reranker score to train data `finetune_data_hn_mined_round2[_test].jsonl`
* prompt: 
Predict whether the passage B explains the mathematical misconception that leads to the wrong answer in query A.
* detailed instruct:
A: {query} B: {passage} {prompt}

cuda:0 -> 19G VRAM 12min 3506 iters
Train data:
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
Test data:
```bash
python add_reranker_score.py \
--input_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round2_test.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round2_test_score.jsonl \
--reranker_name_or_path BAAI/bge-reranker-v2-gemma \
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

## FineTune Embedder with KD
lora r=64,a=32,lr=1e-4, bs=8*2
5 epochs / 1095 iters / 2hr
```bash
sh icl_kd_finetune.sh 2>&1 | tee ./logs/icl_kd_finetune_round1_$(date +%Y%m%d_%H%M%S).log
```
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
|1      | 3.98   |  0.4615    | 0.875/0.9224/0.9444/0.9537 | ? |
|2      | 3.96  |  0.4653    | 0.875/0.9236/0.9386/0.9571 | ? |
|3      | 3.84  |  0.4902    | 0.8842/0.9328/0.9490/0.9629 | ? |
|4      | 3.84  |  0.4921    | 0.8859/0.9344/0.9505/0.9638 | ? |
|5      | 3.83  |  0.4863    | 0.8761/0.9224/0.9490/0.9629 | ? |