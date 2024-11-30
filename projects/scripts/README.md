# LLM Embedder Evaluation (No Finetune)

## Performance Results

### With Examples in Query
```bash
python eval_raw_llm_embedder.py --use_examples_in_query=True
```
| Metric | Score |
|--------|-------|
| MAP@25 | 0.209 |
| Recall@25 | 0.565 |

### Without Examples in Query
```bash
python eval_raw_llm_embedder.py --use_examples_in_query=False
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
## 1. Gen data
```bash
python gen_data_for_hn_mine.py \
    --mode submission
python gen_data_for_hn_mine.py \
    --mode validation
```

## 2. Finetune
```bash
python hn_mine.py \
--embedder_name_or_path BAAI/bge-large-en-v1.5 \
--input_file ../examples/finetune/embedder/bge_finetune_data/finetune_data_submission.jsonl \
--output_file ../examples/finetune/embedder/bge_finetune_data/finetune_data_submission_minedHNjsonl \
--candidate_pool ../examples/finetune/embedder/bge_finetune_data/candidate_pool_submission.jsonl \
--range_for_sampling 2-200 \
--negative_number 15
```