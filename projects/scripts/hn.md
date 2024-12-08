# Hard Negatives mining

# Iteration 0
## HN

### Mine
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
--add_examples_for_task False \
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
--add_examples_for_task False \
--batch_size 1024 \
--embedder_query_max_length 1024 \
--embedder_passage_max_length 512
```


### Finetune

```bash
(sh icl_finetune.sh 2>&1 | tee ./logs/icl_finetune_iter0_hn_$(date +%Y%m%d_%H%M%S).log)               # ; /usr/bin/shutdown
```
lora r=32,a=64,lr=1e-4, bs=8*4
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 1.2708 | 0.4683 | 0.8692/0.9247/0.9502/0.9629 | ? |

<!-- lora r=32,a=64,lr=1e-4, bs=8*2, on mistral-7b [note: add 3 special tokens, emb changed]
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 1     | 0.9872 | 0.4255 | 0.8506/0.9097/0.9421/0.9517 | ? | -->

<!-- lora r=64,a=32,lr=1e-4, bs=8*2
5 epochs / 1095 iters / 2hr

| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
| 0[pretrain] | | 0.2299 | 0.590 | 0.216 |
|1[n=1] | 1.03 | 0.4440 | 0.8483/0.9039/0.9305/0.9432 | ? |
|2[n=2] | 1.04 | 0.4566 | 0.8472/0.9004/0.9270/0.9490 | ? |
|3[n=3] | 0.96 | 0.4890 | 0.8750/0.9224/0.9513/0.9664 | ? |
|4[n=4] | 1.07 | 0.4966 | 0.8750/0.9305/0.9513/0.9548 | ? |
|5[n=5] | 0.88 | 0.5113 | 0.8819/0.9386/0.9537/0.9653 | ? | -->

Choose epoch 1 for next round. Merge model and save.

```bash
python save_merged_model.py \
--base_model_path BAAI/bge-en-icl \
--lora_path ../model_output/icl_finetune_iter0_hn/lora_epoch_1 \
--output_dir ../model_output/icl_finetune_iter0_hn/merged_model_lora_epoch_1
```

Eval & Save retrieval results:
```bash
python eval_llm_embedder.py \
--use_examples_in_query True \
--model_path BAAI/bge-en-icl \
--lora_path ../model_output/icl_finetune_iter0_hn/lora_epoch_1 \
--query_max_len 512 \
--device cuda:0 \
--save_retrieval_results True \
--k 100 \
--retrieval_results_path ../model_output/icl_finetune_iter0_hn/retrieval_results.jsonl
```

# Iteration 1

## Mine 
* range_for_sampling: [2, 200]
* negative_number: 15
Train data:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/icl_finetune_iter0_hn/merged_model \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_hn.jsonl \
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
--embedder_name_or_path ../model_output/icl_finetune_iter0_hn/merged_model \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_test_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_iter1_hn_test.jsonl \
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
## Finetune
lora r=32,a=64,lr=1e-4, bs=8*4
```bash
sh icl_finetune.sh 2>&1 | tee ./logs/icl_finetune_iter1_hn_$(date +%Y%m%d_%H%M%S).log
```
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
|1      | 1.4966 | 0.4423 | 0.8622/0.9212/0.9490/0.9560 | ？ |
|2      | 1.4050 | 0.4994 | 0.8854/0.9259/0.9444/0.9629 | 0.365 |
|3      | 1.4305 | 0.5224 | 0.8904/0.9247/0.9525/0.9618 | ? |
<!-- 
| Epoch | eval_loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|--------|-----------|---------|---------|
|1      | 1.36   |  0.4595    | 0.8553/0.9027/0.9305/0.9409 | ? |
|2      | 1.30  |  0.4756    | 0.8645/0.9212/0.9421/0.9537 | 0.366 |
|3      | 1.38   |  0.5035    | 0.8715/0.9203/0.9537/0.9652 | ? |
|4      | 1.45   |  0.5189    | 0.875/0.9270/0.9490/0.9629 | ? | -->

Choose Epoch 2 for next round. Merge model and save.
```bash
python save_merged_model.py \
--base_model_path BAAI/bge-en-icl \
--lora_path ../model_output/icl_finetune_iter1_hn/lora_epoch_2 \
--output_dir ../model_output/icl_finetune_iter1_hn/merged_model_lora_epoch_2
```

Eval & Save retrieval results:
`Saving retrieval results to ../model_output/icl_finetune_iter1_hn/retrieval_results.jsonl`
```bash
python eval_llm_embedder.py \
--use_examples_in_query True \
--model_path BAAI/bge-en-icl \
--lora_path ../model_output/icl_finetune_iter1_hn/lora_epoch_2 \
--query_max_len 512 \
--device cuda:0 \
--save_retrieval_results True \
--k 100 \
--retrieval_results_path ../model_output/icl_finetune_iter1_hn/retrieval_results.jsonl
```
