# Iter-0

# Submission: No eval needed / modify reranker_finetune.sh and icl_finetune.sh: remove all eval data inputs

## 1. Hard negative mining using bge-en-icl
* range_for_sampling: [2, 200]
* negative_number: 15

Train data:
[DONE]
```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-en-icl \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/submission/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/submission/finetune_data_iter0_hn.jsonl \
--candidate_pool ../data/embedder_train_eval_data/submission/corpus.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:0 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task True \
--batch_size 1024 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```

## 2. Finetune Qwen2.5-14B-Instruct with Hard negative mining results [Using ICL]
[DONE]
3 epochs for sub / 2 gpus / ga = 1 / lr = 1e-4 / total_bs=16 -> 1e-4
```bash
sh icl_finetune.sh \
--knowledge_distillation False \
--epochs 3 \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--num_gpus 2 \
--learning_rate 1e-4 \
--gpu_ids "0,1" \
--train_data ../data/embedder_train_eval_data/submission/finetune_data_iter0_hn.jsonl \
--output_dir ../model_output/submission/embedder_icl_finetune_qwen14b_iter0 \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
| Epoch | eval_loss | MAP@25 | Recall@25 | LB Score |
|-------|--------|-----------|---------|---------|

## 3. HN mine using Finetuned model
* range_for_sampling: [2, 100]
* negative_number: 15
[DONE]
Train:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/submission/embedder_icl_finetune_qwen14b_iter0/merged_model \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/submission/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/submission/finetune_data_iter0_hn_from_qwen14b_iter0_for_ranker.jsonl \
--candidate_pool ../data/embedder_train_eval_data/submission/corpus.jsonl \
--range_for_sampling 2-100 \
--negative_number 15 \
--devices cuda:0 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task True \
--batch_size 256 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```

## 5. Finetune reranker [Qwen2.5-14B-Instruct-finetuned] using hard negative mine by finetuned embedder [4hr]
bs1*ga8*n4 lr=2e-4; 40G 显存, 3.5hr for 4epochs [sample eval files 0.4 to avoid NCCL timeout]
Early stop after 3 epochs, path: `../model_output/submission/reranker_finetune_qwen14b_iter0/lora_epoch_2`
[DONE]
```bash
sh reranker_finetune.sh \
--epochs 4 \
--batch_size 1 \
--gradient_accumulation_steps 8 \
--num_gpus 4 \
--learning_rate 2e-4 \
--gpu_ids "0,1,2,3" \
--train_data ../data/embedder_train_eval_data/submission/finetune_data_iter0_hn_from_qwen14b_iter0_for_ranker.jsonl \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct \
--output_dir ../model_output/submission/reranker_finetune_qwen14b_iter0
```
| Epoch | eval_loss | MAP@25 | Recall@25 | LB Score |
|-------|--------|-----------|---------|---------|

Merge reranker
[DONE]
```bash
python save_merged_model.py \
--base_model_path Qwen/Qwen2.5-14B-Instruct \
--type reranker \
--lora_path ../model_output/submission/reranker_finetune_qwen14b_iter0/lora_epoch_2 \
--output_dir ../model_output/submission/reranker_finetune_qwen14b_iter0/merged_model
```


# Iter-1
## 1. Hard negative mining using `embedder_icl_finetune_qwen14b_iter0`
* range_for_sampling: [2, 200]
* negative_number: 15
[DONE]
Train:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/submission/embedder_icl_finetune_qwen14b_iter0/merged_model \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/submission/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/submission/finetune_data_iter1_hn.jsonl \
--candidate_pool ../data/embedder_train_eval_data/submission/corpus.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:0 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task True \
--batch_size 256 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```

## 2. Add ranker score to `finetune_data_iter1_hn`
[DONE]
Train data:[1hr]
```bash
python add_reranker_score.py \
--input_file ../data/embedder_train_eval_data/submission/finetune_data_iter1_hn.jsonl \
--output_file ../data/embedder_train_eval_data/submission/finetune_data_iter1_hn_scored.jsonl \
--reranker_name_or_path ../model_output/submission/reranker_finetune_qwen14b_iter0/merged_model \
--reranker_model_class decoder-only-base \
--query_instruction_for_rerank 'A: ' \
--query_instruction_format '{}{}' \
--passage_instruction_for_rerank 'B: ' \
--passage_instruction_format '{}{}' \
--prompt 'Given a query A and a passage B, determine whether the passage B explains the mathematical misconception that leads to the wrong answer in query A by providing a prediction of either "Yes" or "No".' \
--reranker_max_length 512 \
--use_bf16 True \
--reranker_batch_size 8 \
--devices cuda:0
```

## 3. Finetune Qwen2.5-14B-Instruct with Hard negative mining results [Using embedder iter 0]
Use Teacher Scores:
4 epochs / 2 gpus / ga = 1 / lr = 1e-4 / total_bs=16 -> 1e-4 [2hr]

epoch 3 is OK, let's see if epoch 4 looks good, loss not decreasing in ep4, take epoch 3.

```bash
sh icl_finetune.sh \
--knowledge_distillation True \
--epochs 4 \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--num_gpus 2 \
--learning_rate 1e-4 \
--gpu_ids "0,1" \
--train_data ../data/embedder_train_eval_data/submission/finetune_data_iter1_hn_scored.jsonl \
--output_dir ../model_output/submission/embedder_icl_finetune_qwen14b_iter1_with_kd \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```

Merge embedder
[DONE]
```bash
python save_merged_model.py \
--base_model_path Qwen/Qwen2.5-14B-Instruct \
--type embedder \
--lora_path ../model_output/submission/embedder_icl_finetune_qwen14b_iter1_with_kd/lora_epoch_3rd \
--output_dir ../model_output/submission/embedder_icl_finetune_qwen14b_iter1_with_kd/merged_model
```

## 4. HN mine using Finetuned model [embedder iter 1]
* range_for_sampling: [2, 100]
* negative_number: 15
Train:
[DONE]
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/submission/embedder_icl_finetune_qwen14b_iter1_with_kd/merged_model \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/submission/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/submission/finetune_data_iter1_hn_from_qwen14b_iter1_with_kd_for_ranker.jsonl \
--candidate_pool ../data/embedder_train_eval_data/submission/corpus.jsonl \
--range_for_sampling 2-100 \
--negative_number 15 \
--devices cuda:0 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task True \
--batch_size 256 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```

## 5. Add score for `finetune_data_iter0_hn_from_qwen14b_iter0_for_ranker` using reranker iter 0
Train data: [1.5hr]
```bash
python add_reranker_score.py \
--input_file ../data/embedder_train_eval_data/submission/finetune_data_iter1_hn_from_qwen14b_iter1_with_kd_for_ranker.jsonl \
--output_file ../data/embedder_train_eval_data/submission/finetune_data_iter1_hn_from_qwen14b_iter1_with_kd_for_ranker_scored.jsonl \
--reranker_name_or_path ../model_output/submission/reranker_finetune_qwen14b_iter0/merged_model \
--reranker_model_class decoder-only-base \
--query_instruction_for_rerank 'A: ' \
--query_instruction_format '{}{}' \
--passage_instruction_for_rerank 'B: ' \
--passage_instruction_format '{}{}' \
--prompt 'Given a query A and a passage B, determine whether the passage B explains the mathematical misconception that leads to the wrong answer in query A by providing a prediction of either "Yes" or "No".' \
--reranker_max_length 512 \
--use_bf16 True \
--reranker_batch_size 4 \
--devices cuda:0
```

## 5. Finetune reranker [Qwen2.5-14B-Instruct] using hard negative mine by finetuned embedder v1 [4hr]
bs1*ga8*n4 lr=2e-4; 40G 显存, 3.5hr for 4epochs [sample eval files 0.4 to avoid NCCL timeout]
[consider early stop at epoch 2/3 -> 1.7hr(100min)]

Choose Epoch 3, `projects/model_output/submission/reranker_finetune_qwen14b_iter1_with_kd/lora_epoch_3`

With Teacher Scores: [4hr]
```bash
sh reranker_finetune.sh \
--knowledge_distillation True \
--label_smoothing 0.2 \
--epochs 4 \
--batch_size 1 \
--gradient_accumulation_steps 8 \
--num_gpus 4 \
--learning_rate 2e-4 \
--gpu_ids "0,1,2,3" \
--train_data ../data/embedder_train_eval_data/submission/finetune_data_iter1_hn_from_qwen14b_iter1_with_kd_for_ranker_scored.jsonl \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct \
--output_dir ../model_output/submission/reranker_finetune_qwen14b_iter1_with_kd
```

# CV / LB
iter0: cv 0.548 / lb 0.502
iter1: cv 0.591 / lb 0.505