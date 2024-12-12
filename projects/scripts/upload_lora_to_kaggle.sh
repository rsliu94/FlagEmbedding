kaggle datasets init -p projects/model_output/cross_validation/reranker_finetune_qwen14b_iter0/lora_epoch_2
kaggle datasets create -p projects/model_output/cross_validation/reranker_finetune_qwen14b_iter0/lora_epoch_2
kaggle datasets version -p projects/model_output/cross_validation/reranker_finetune_qwen14b_iter0/lora_epoch_2  -m "1st Version"



submission-embedder-iter0 [live]
[DONE]
kaggle datasets init -p projects/model_output/submission/embedder_icl_finetune_qwen14b_iter0/lora_epoch_3
kaggle datasets create -p projects/model_output/submission/embedder_icl_finetune_qwen14b_iter0/lora_epoch_3

submission-reranker-iter0 [live]
[DONE]
kaggle datasets init -p projects/model_output/submission/reranker_finetune_qwen14b_iter0/lora_epoch_2
kaggle datasets create -p projects/model_output/submission/reranker_finetune_qwen14b_iter0/lora_epoch_2

submission-embedder-iter1 [live]
[DONE]
kaggle datasets init -p projects/model_output/submission/embedder_icl_finetune_qwen14b_iter1_with_kd/lora_epoch_3rd
kaggle datasets create -p projects/model_output/submission/embedder_icl_finetune_qwen14b_iter1_with_kd/lora_epoch_3rd

submission-reranker-iter1

kaggle datasets init -p projects/model_output/submission/reranker_finetune_qwen14b_iter1_with_kd/lora_epoch_3
kaggle datasets create -p projects/model_output/submission/reranker_finetune_qwen14b_iter1_with_kd/lora_epoch_3


cv-embedder-iter1 [live]
[DONE]
kaggle datasets init -p projects/model_output/cross_validation/embedder_icl_finetune_qwen14b_iter1_with_kd/lora_epoch_3
kaggle datasets create -p projects/model_output/cross_validation/embedder_icl_finetune_qwen14b_iter1_with_kd/lora_epoch_3

cv-reranker-iter1-with-kd [live] [but need to compare with w/o kd][go with kd]
[DONE]
kaggle datasets init -p projects/model_output/cross_validation/reranker_finetune_qwen14b_iter1_with_kd/lora_epoch_3rd
kaggle datasets create -p projects/model_output/cross_validation/reranker_finetune_qwen14b_iter1_with_kd/lora_epoch_3rd