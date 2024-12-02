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

# Hard Negative Mining
## 1. Gen input data
```bash
python gen_data_for_hn_mine.py \
    --mode submission
python gen_data_for_hn_mine.py \
    --mode validation
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
| MAP@25 | 0.4717764314771157 |
| Recall@25 | 0.8767416934619507 |
