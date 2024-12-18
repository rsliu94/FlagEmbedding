# LLM Embedder Evaluation (No Finetune)

## Performance Results

### With Examples in Query
```bash
python eval_llm_embedder.py --use_examples_in_query=True
```
| Metric | Score |
|--------|-------|
| MAP@25 | 0.209 |
| Recall@25 | 0.565 |

### Without Examples in Query
```bash
python eval_llm_embedder.py --use_examples_in_query=False
```
| Metric | Score |
|--------|-------|
| MAP@25 | 0.196 |
| Recall@25 | 0.550 |

## 结论
使用示例查询（`use_examples_in_query=True`）可以获得更好的性能：
- MAP@25 提升了 6.4%
- Recall@25 提升了 2.8%

# CV vs LB investigation
## No LoRA, on v2 validation set, on kaggle
CV = 0.2089; `submission_no_lora_val_v2.csv`
LB = 0.211
## on local machine
```bash
python eval_llm_embedder.py \
    --model_path=BAAI/bge-en-icl \
    --use_examples_in_query=True \
    --validation_version=2
```
CV = 0.2093; download & compare with kaggle results
1. submission.csv is different
2. corpus/queries list are the same
3. doc/query embeddings are different

## Use LoRA, on v2 validation set, on kaggle
CV = 0.4714;
LB = 0.304
## on local machine
```bash
python eval_llm_embedder.py \
    --model_path=BAAI/bge-en-icl \
    --use_examples_in_query=True \
    --lora_path=../model_output/icl_finetune_validation_v2_round1/lora_epoch_5 \
    --validation_version=2
```
CV = 0.4709;

# Hard Negative Mining
## 1. Gen input data
```bash
python gen_data_for_hn_mine.py \
    --mode submission \
    --sub_dir_name hn_mine_data_zero_round
python gen_data_for_hn_mine.py \
    --mode validation \
    --sub_dir_name hn_mine_data_zero_round
```

## 2. Mine hard negative use icl model
```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-en-icl \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/hn_mine_data_zero_round/finetune_data_validation.jsonl \
--output_file ../data/hn_mine_data_zero_round/finetune_data_validation_minedHN.jsonl \
--candidate_pool ../data/hn_mine_data_zero_round/candidate_pool_validation.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:0
```

## 3. Finetune ICL with mined hard negative
```bash
./icl_finetune.sh
```
## 4. eval finetuned model
```bash
python eval_llm_embedder.py \
    --model_path ../model_output/test_decoder_only_base_bge-en-icl_sd \
    --use_examples_in_query=True
```
| Metric | Score |
|--------|-------|
| MAP@25 | 0.4117 |
| Recall@25 | 0.8676 |

## 5. LoRA Finetune ICL 多轮训练结果

[完整实验记录(Wandb)](https://wandb.ai/rsliu94/huggingface/runs/mbhmntcf/workspace?nw=nwuserrsliu94)

### 训练参数
| 参数 | 值 |
|------|-----|
| 训练轮数 | 5 |
| 批次大小 | 16 |
| 学习率 | 1e-4 |
| GPU数量 | 2 |
| 每轮训练时间 | 10分钟 |
| 每轮评估时间 | 3分钟 |

| Epoch | MAP@25 | Recall@25 |
|-------|--------|-----------|
| 1 | 0.3954 | 0.8376 |
| 2 | 0.4369 | 0.8655 |
| 3 | 0.4468 | 0.8735 |
| 4 | 0.4685 | 0.8735 |
| 5 | 0.4716 | 0.8767 |

```bash
# doublecheck eval result at epoch 4
python eval_llm_embedder.py \
    --lora_path ../model_output/icl_finetune_round1/lora_epoch_4 \
    --use_examples_in_query=True
```
| Metric | Score |
|--------|-------|
| MAP@25 | 0.47177 |
| Recall@25 | 0.8767 |
With BnB config:
| Metric | Score |
|--------|-------|
| MAP@25 | 0.4674|
| Recall@25 | 0.8762 |

```bash
# doublecheck eval result at epoch 5
python eval_llm_embedder.py \
    --lora_path ../model_output/icl_finetune_round1/lora_epoch_5 \
    --use_examples_in_query=True
```
| Metric | Score |
|--------|-------|
| MAP@25 | 0.4710 |
| Recall@25 | 0.8815 |
```bash
# check eval result at epoch 4 with v1 validation set
python eval_llm_embedder.py \
    --lora_path ../model_output/icl_finetune_round1/lora_epoch_4 \
    --use_examples_in_query=True \
    --validation_version=1
```
This is polluted because the training set is from validation_v2 split. Need to conduct training on validation_v1 split...
| Metric | Score |
|--------|-------|
| MAP@25 |  0.6766 |
| Recall@25 | 0.9431 |

**Epoch 5 LB score: 0.304**

# Repeat FineTune ICL with validation_v1 split
```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-en-icl \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/hn_mine_data_zero_round/validation_v1/finetune_data.jsonl \
--output_file ../data/hn_mine_data_zero_round/validation_v1/finetune_data_minedHN.jsonl \
--candidate_pool ../data/hn_mine_data_zero_round/validation_v1/candidate_pool.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:0
```
| Epoch | MAP@25 | Recall@25 |
|-------|--------|-----------|
| 1 | 0.3920 | 0.8321 |
| 2 | 0.4411 | 0.8616 |
| 3 | 0.4574 | 0.8561 |
| 4 | 0.4688 | 0.8732 |
| 5 | 0.4764 | 0.8575 |
```bash
# local eval at epoch 4
python eval_llm_embedder.py \
    --lora_path ../model_output/icl_finetune_validation_v1_round1/lora_epoch_4 \
    --use_examples_in_query=True \
    --validation_version=1
```
| Metric | Score |
|--------|-------|
| MAP@25 | 0.4717 |
| Recall@25 | 0.8773 |

Use epoch 4 for lb score.
**LB Score: 0.308**

# Repeat ICL Finetune with whole train split [submission folder]
```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-en-icl \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/hn_mine_data_zero_round/submission/finetune_data.jsonl \
--output_file ../data/hn_mine_data_zero_round/submission/finetune_data_minedHN.jsonl \
--candidate_pool ../data/hn_mine_data_zero_round/submission/candidate_pool.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:0
```
**LB Score: 0.319**

# Repeat ICL Finetune with validation_v3 split
```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-en-icl \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/hn_mine_data_zero_round/validation_v3/finetune_data.jsonl \
--output_file ../data/hn_mine_data_zero_round/validation_v3/finetune_data_minedHN.jsonl \
--candidate_pool ../data/hn_mine_data_zero_round/validation_v3/candidate_pool.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:0
```
| Epoch | MAP@25 | Recall@25 |
|-------|--------|-----------|
| 1 | 0.4292 | 0.8483 |
| 2 | 0.4388 | 0.8437 |
| 3 | 0.4726 | 0.8576 |
| 4 | 0.4754 | 0.8460 |
| 5 | 0.4877 | 0.8506 |
**LoRA epoch 1 LB Score: 0.316**
**LoRA epoch 3 LB Score: 0.318**

# Repeat ICL Finetune with validation_v3 split + shuffle minedHN
```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-en-icl \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/hn_mine_data_zero_round/validation_v3/finetune_data.jsonl \
--output_file ../data/hn_mine_data_zero_round/validation_v3/finetune_data_minedHN.jsonl \
--candidate_pool ../data/hn_mine_data_zero_round/validation_v3/candidate_pool.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--shuffle_data True \
--devices cuda:0
```
| Epoch | MAP@25 | Recall@25 |
|-------|--------|-----------|
| 1 | 0.3797 | 0.8090 |
| 2 | 0.4559 | 0.8564 |
| 3 | 0.4510 | 0.8414 |
| 4 | 0.4773 | 0.8680 |
| 5 | 0.4800 | 0.8680 |
**LoRA epoch 1 LB Score: 0.303**

With out examples in query
| Epoch | MAP@25 | Recall@25 |
|-------|--------|-----------|
| 1 | 0.3717 | 0.7928 |
| 2 | 0.4449 | 0.8530 |
| 3 | 0.4710 | 0.8645 |
| 4 | 0.4733 | 0.8692 |

# Repeat ICL Finetune with validation_v4 split + shuffle minedHN
```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-en-icl \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/hn_mine_data_zero_round/validation_v4/finetune_data.jsonl \
--output_file ../data/hn_mine_data_zero_round/validation_v4/finetune_data_minedHN.jsonl \
--candidate_pool ../data/hn_mine_data_zero_round/validation_v4/candidate_pool.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--shuffle_data True \
--devices cuda:0
```
| Epoch | MAP@25 | Recall@25 |
|-------|--------|-----------|
| 1 | 0.5119 | 0.9050 |
| 2 | 0.6285 | 0.9270 |

# Iterative Hard Negative Mining

## Round 0, New Validation Set + Data Pipeline
`Per-batch-size=8`
| Epoch | MAP@25 | Recall@25 | LB Score |
|-------|--------|-----------|---------|
| 0[pretrain] | 0.2299 | 0.590 | 0.216 |
| 0[augment-examples-n=1] | 0.231 | 0.581 | 0.221 |
| 0[augment-examples-n=2] | 0.233 | 0.575 | ? |
| 1 | 0.4452 | 0.8530 | 0.321 |

### Merge lora with base model
```bash
python save_merged_model.py \
    --base_model_path BAAI/bge-en-icl \
    --lora_path ../model_output/icl_finetune_round1/lora_epoch_1 \
    --output_dir ../model_output/icl_finetune_round1/merged_model_lora_epoch_1
```
### Mine hard negative use FineTuned icl model [Round 1]
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/icl_finetune_round1/merged_model_lora_epoch_1 \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round1.jsonl \
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
| Epoch | MAP@25 | Recall@25 | LB Score |
|-------|--------|-----------|---------|
| 1[augment-examples-n=1] | 0.4484 | 0.8645 | 0.367 |
| 2[augment-examples-n=1] | 0.5087 | 0.8888 | 0.388 |
```bash
python eval_llm_embedder.py \
--lora_path ../model_output/icl_finetune_round2/lora_epoch_2 \
--use_examples_in_query=True \
--num_examples=2 \
--query_max_len=1024
```

### Mine hard negative use FineTuned icl model [Round 2]
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/icl_finetune_round2/merged_model \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round2.jsonl \
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
| Epoch | MAP@25 | Recall@25 | LB Score |
|-------|--------|-----------|---------|
| 1[augment-examples-n=1] | 0.4621 | 0.8784 | ? |
| 2[augment-examples-n=1] | 0.4938 (0.50) | 0.8912 | 0.363 |
Eval with script: 0.50
```bash
python eval_llm_embedder.py \
--lora_path ../model_output/icl_finetune_round3/lora_epoch_2 \
--use_examples_in_query=True \
--num_examples=1 \
--query_max_len=512
```

### Mine hard negative use FineTuned icl model [Round 3]
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/icl_finetune_round3/merged_model \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round3.jsonl \
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
For test data:
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/icl_finetune_round3/merged_model \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/embedder_train_eval_data/cross_validation/hn_mine_test_input.jsonl \
--output_file ../data/embedder_train_eval_data/cross_validation/finetune_data_hn_mined_round3_test.jsonl \
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
### 训练配置优化

#### 参数调整
| 参数 | 调整前 | 调整后 |
|------|--------|--------|
| 查询示例数量上限 | 6 | 3 |
| 示例最大长度 | 512 | 384 |
| 查询最大长度 | 2048 | 1024 |

#### LoRA 参数设置
当前配置:
- LoRA rank: 32
- LoRA alpha: 64

论文建议配置(try it later):
- LoRA rank: 64 
- LoRA alpha: 32
- 学习率: 1e-4

| Epoch | Eval Loss | MAP@25 | Recall@25/50/75/100 | LB Score |
|-------|-----------|--------|-----------|---------|
| 0     | 2.77      |        |           |        |
| 1[augment-examples-n=1] | 1.255 | 0.4395  | 0.8657/0.9178/0.9317/0.9537  |  0.321  |
| 2[augment-examples-n=1] | 1.2373 | 0.4910  | 0.8761/0.9236/0.9479/0.9583  |  0.360  |
| 3[augment-examples-n=1] | 1.1455 | 0.4912  | 0.8842/0.9212/0.9398/0.9571  |  0.354  |
| 4[augment-examples-n=1] | 1.2431 | 0.5128  | 0.8807/0.9247/0.9513/0.9618  |  0.364  |
| 5[augment-examples-n=1] | 1.2363 | 0.5179  | 0.8807/0.9259/0.9513/0.9606  |  0.378  |
```bash
python eval_llm_embedder.py \
--lora_path ../model_output/icl_finetune_round4/lora_epoch_1 \
--use_examples_in_query=True \
--query_max_len=512
```