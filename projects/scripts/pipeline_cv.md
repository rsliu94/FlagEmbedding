# Iter-0

## 1. Hard negative mining using bge-en-icl
* range_for_sampling: [2, 200]
* negative_number: 15

Train data:
```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-en-icl \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:1 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task True \
--batch_size 1024 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```

For test data:
```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-en-icl \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_test_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
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
2 epochs / 2 gpus / ga = 1 / lr = 1e-4 / total_bs=16 -> 1e-4
```bash
sh icl_finetune.sh \
--epochs 2 \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--num_gpus 2 \
--learning_rate 1e-4 \
--gpu_ids "0,1" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl \
--output_dir ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0 \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 0.8871 | 0.4547 | 0.8784/0.9247/0.9375/0.9560 | ? |
| 2     | 0.8026 | 0.4853 | 0.9016/0.9363/0.9490/0.9606 | 0.422 |

Merge lora [for hn mine]
```bash
python save_merged_model.py \
--base_model_path Qwen/Qwen2.5-14B-Instruct \
--lora_path ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0/lora_epoch_2 \
--output_dir ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0/merged_model
```

## 3. HN mine using Finetuned model
* range_for_sampling: [2, 100]
* negative_number: 15

Train:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0/merged_model \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_from_qwen14b_iter0_for_ranker.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
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
Test:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0/merged_model \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_test_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_from_qwen14b_iter0_for_ranker_test.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-100 \
--negative_number 15 \
--devices cuda:1 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task True \
--batch_size 256 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```
## 4. Retrieval & Eval using Qwen2.5-7B-Instruct-finetuned
```bash
python eval_llm_embedder.py \
--use_examples_in_query True \
--model_path ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0/merged_model \
--query_max_len 512 \
--use_examples_in_query True \
--device cuda:0 \
--save_retrieval_results True \
--k 25 \
--batch_size 8 \
--device cuda:2 \
--retrieval_results_path ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0/retrieval_results_top25_for_ranker_test.jsonl
```
map@25_score: 0.48167870154282266
recall@25_score: 0.9027777777777778
recall@50_score: 0.9293981481481481
recall@75_score: 0.9432870370370371
recall@100_score: 0.9652777777777778

## 5. Finetune reranker [Qwen2.5-14B-Instruct-finetuned] using hard negative mine by finetuned embedder [3.5hr]
bs1*ga8*n4 lr=2e-4; 40G 显存, 3.5hr for 4epochs [sample eval files 0.4 to avoid NCCL timeout]
[consider early stop at epoch 2/3 -> 1.7hr(100min)]
```bash
sh reranker_finetune.sh \
--epochs 4 \
--batch_size 1 \
--gradient_accumulation_steps 8 \
--num_gpus 4 \
--learning_rate 2e-4 \
--gpu_ids "0,1,2,3" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_from_qwen14b_iter0_for_ranker.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_from_qwen14b_iter0_for_ranker_test.jsonl \
--eval_retrieval_result_path ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0/retrieval_results_top25_for_ranker_test.jsonl \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct \
--output_dir ../model_output/cross_validation/reranker_finetune_qwen14b_iter0
```
| Epoch | eval_loss | MAP@25(rerank/recall) | Recall@25(rerank/recall) | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 0.8703 | 0.5599/0.5080 | 0.9043/0.9043 | ? |
| 2     | 1.0203 | 0.6190/0.5080 | 0.9043/0.9043 | ? | [early stop, train loss ~ 0.01]

## 6. Eval reranker
Merge
```bash
python save_merged_model.py \
--base_model_path Qwen/Qwen2.5-14B-Instruct \
--type reranker \
--lora_path ../model_output/cross_validation/reranker_finetune_qwen14b_iter0/lora_epoch_2 \
--output_dir ../model_output/cross_validation/reranker_finetune_qwen14b_iter0/merged_model
```
Eval [25min]
```bash
python eval_llm_reranker.py \
--retrieval_results_path ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0/retrieval_results_top25_for_ranker_test.jsonl \
--model_path Qwen/Qwen2.5-14B-Instruct \
--lora_path ../model_output/cross_validation/reranker_finetune_qwen14b_iter0/lora_epoch_2 \
--batch_size 4 \
--query_max_len 512 \
--doc_max_len 128 \
--device cuda:0
```
==Rerank==
map@25_score: 0.5441250714745003
recall@25_score: 0.9027777777777778
==Recall==
map@25_score: 0.48167870154282266
recall@25_score: 0.9027777777777778

Double check [some problem with lm_heads?]

Some weights of Qwen2ForCausalLM were not initialized from the model checkpoint at ../model_output/cross_validation/reranker_finetune_qwen14b_iter0/merged_model and are newly initialized: ['lm_head.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

```bash
python eval_llm_reranker.py \
--retrieval_results_path ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0/retrieval_results_top25_for_ranker_test.jsonl \
--model_path ../model_output/cross_validation/reranker_finetune_qwen14b_iter0/merged_model \
--batch_size 4 \
--query_max_len 512 \
--doc_max_len 128 \
--device cuda:1
```
==Rerank==
map@25_score: 0.14888041588262946
recall@25_score: 0.9027777777777778
==Recall==
map@25_score: 0.48167870154282266
recall@25_score: 0.9027777777777778

Fix and try eval again
==Rerank==
map@25_score: 0.5485555580229804
recall@25_score: 0.9027777777777778
==Recall==
map@25_score: 0.48167870154282266
recall@25_score: 0.9027777777777778


# Iter-1
## 1. Hard negative mining using `embedder_icl_finetune_qwen14b_iter0`
* range_for_sampling: [2, 200]
* negative_number: 15

Train:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0/merged_model \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_hn.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:2 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task True \
--batch_size 256 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```
Test:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter0/merged_model \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_test_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_hn_test.jsonl \
--candidate_pool ../data/embedder_train_eval_data/cross_validation/corpus.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:3 \
--shuffle_data True \
--query_instruction_for_retrieval "Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer." \
--query_instruction_format '<instruct>{}\n<query>{}' \
--add_examples_for_task True \
--batch_size 256 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```

## 2. Add ranker score to `finetune_data_iter1_hn`
Train data: [1hr?]
```bash
python add_reranker_score.py \
--input_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_hn.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_hn_scored.jsonl \
--reranker_name_or_path Qwen/Qwen2.5-14B-Instruct \
--reranker_model_class decoder-only-base \
--reranker_peft_path ../model_output/cross_validation/reranker_finetune_qwen14b_iter0/lora_epoch_2 \
--query_instruction_for_rerank 'A: ' \
--query_instruction_format '{}{}' \
--passage_instruction_for_rerank 'B: ' \
--passage_instruction_format '{}{}' \
--prompt 'Given a query A and a passage B, determine whether the passage B explains the mathematical misconception that leads to the wrong answer in query A by providing a prediction of either "Yes" or "No".' \
--reranker_max_length 512 \
--use_bf16 True \
--reranker_batch_size 4 \
--devices cuda:2
```
Test data: [15min]
```bash
python add_reranker_score.py \
--input_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_hn_test.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_hn_test_scored.jsonl \
--reranker_name_or_path ../model_output/cross_validation/reranker_finetune_qwen14b_iter0/merged_model \
--reranker_model_class decoder-only-base \
--query_instruction_for_rerank 'A: ' \
--query_instruction_format '{}{}' \
--passage_instruction_for_rerank 'B: ' \
--passage_instruction_format '{}{}' \
--prompt 'Given a query A and a passage B, determine whether the passage B explains the mathematical misconception that leads to the wrong answer in query A by providing a prediction of either "Yes" or "No".' \
--reranker_max_length 512 \
--use_bf16 True \
--reranker_batch_size 4 \
--devices cuda:3
```

## 3. Finetune Qwen2.5-14B-Instruct with Hard negative mining results [Using embedder iter 0]
Use Teacher Scores or Not?

Use Teacher Scores:

3 epochs / 2 gpus / ga = 1 / lr = 1e-4 / total_bs=16 -> 1e-4
```bash
sh icl_finetune.sh \
--knowledge_distillation True \
--epochs 3 \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--num_gpus 2 \
--learning_rate 1e-4 \
--gpu_ids "0,1" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_hn_scored.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_hn_test_scored.jsonl \
--output_dir ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter1_with_kd \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 1.5677 | 0.4646 | 0.8865/0.9270/0.9513/0.9652 | ? |
| 2     | 1.3430 | 0.5169 | 0.9108/0.9432/0.9594/0.9687 | ? |

No Teacher Scores:

2 epochs / 2 gpus / ga = 1 / lr = 1e-4 / total_bs=16 -> 1e-4
```bash
sh icl_finetune.sh \
--knowledge_distillation False \
--epochs 2 \
--batch_size 8 \
--gradient_accumulation_steps 1 \
--num_gpus 2 \
--learning_rate 1e-4 \
--gpu_ids "2,3" \
--train_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_hn_scored.jsonl \
--eval_data ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_hn_test_scored.jsonl \
--output_dir ../model_output/cross_validation/embedder_icl_finetune_qwen14b_iter1_without_kd \
--model_name_or_path Qwen/Qwen2.5-14B-Instruct
```
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 1.0706 | 0.4589 | 0.8842/0.9328/0.9502/0.9629 | ? |
| 2     | 0.9785 | 0.4934 | 0.9050/0.9444/0.9560/0.9699 | ? |