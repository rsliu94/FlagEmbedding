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




# Iterative Hard Negative Mining
## 1. Hard Negative Mining, Round 1
### 1. Gen input data
```bash
python gen_data_for_hn_mine.py \
    --mode submission \
    --sub_dir_name hn_mine_data_round_1
python gen_data_for_hn_mine.py \
    --mode validation \
    --sub_dir_name hn_mine_data_round_1
```
### 2. Mine hard negative use FineTuned icl model
`projects/model_output/icl_finetune_round1/checkpoint-390`
```bash
python hn_mine.py \
--embedder_name_or_path ../model_output/icl_finetune_round1/checkpoint-390 \
--embedder_model_class decoder-only-icl \
--pooling_method last_token \
--input_file ../data/hn_mine_data_round_1/finetune_data_validation.jsonl \
--output_file ../data/hn_mine_data_round_1/finetune_data_validation_minedHN.jsonl \
--candidate_pool ../data/hn_mine_data_round_1/candidate_pool_validation.jsonl \
--range_for_sampling 2-200 \
--negative_number 15 \
--devices cuda:0
```
### 3. continue finetune ICL with mined hard negative in round 1
```bash
./icl_finetune.sh
```
