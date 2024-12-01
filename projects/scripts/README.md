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