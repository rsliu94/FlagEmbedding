# export WANDB_MODE=disabled
# sh reranker_finetune.sh --epochs 1 --batch_size 1 --num_gpus 2 --gpu_ids "0,1" 2>&1 | tee ./logs/reranker_finetune_iter1_$(date +%Y%m%d_%H%M%S).log
# sh reranker_finetune.sh --epochs 4 --batch_size 2 --num_gpus 1 --gpu_ids "0"
# sh reranker_finetune.sh --epochs 5 --batch_size 1 --num_gpus 4 --gpu_ids "0,1,2,3" 2>&1 | tee ./logs/reranker_finetune_qwen_$(date +%Y%m%d_%H%M%S).log && /usr/bin/shutdown
source /etc/network_turbo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 打印初始参数
echo "输入参数: $@"

# 添加参数解析
while [[ $# -gt 0 ]]; do
  key="$1"
  echo "处理参数: $key"
  
  case $key in
    --epochs)
      num_train_epochs="$2"
      echo "设置 epochs = $num_train_epochs"
      shift 2
      ;;
    --batch_size)
      per_device_train_batch_size="$2"
      echo "设置 batch_size = $per_device_train_batch_size"
      shift 2
      ;;
    --num_gpus)
      num_gpus="$2"
      echo "设置 num_gpus = $num_gpus"
      shift 2
      ;;
    --gpu_ids)
      export CUDA_VISIBLE_DEVICES="$2"
      echo "设置 CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
      shift 2
      ;;
    --train_data)
      train_data="$2"
      echo "设置 train_data = $train_data"
      shift 2
      ;;
    --eval_data)
      eval_data="$2"
      echo "设置 eval_data = $eval_data"
      shift 2
      ;;
    --model_name_or_path)
      model_name_or_path="$2"
      echo "设置 model_name_or_path = $model_name_or_path"
      shift 2
      ;;
    --output_dir)
      output_dir="$2"
      echo "设置 output_dir = $output_dir"
      shift 2
      ;;
    --gradient_accumulation_steps)
      gradient_accumulation_steps="$2"
      echo "设置 gradient_accumulation_steps = $gradient_accumulation_steps"
      shift 2
      ;;
    *)
      echo "错误: 未知参数 '$key'"
      echo "可用参数: --epochs, --batch_size, --num_gpus, --gpu_ids, --train_data, --eval_data, --model_name_or_path, --output_dir, --gradient_accumulation_steps"
      exit 1
      ;;
  esac
done

# 打印最终设置的值
echo "最终配置:"
echo "num_train_epochs = $num_train_epochs"
echo "per_device_train_batch_size = $per_device_train_batch_size"
echo "num_gpus = $num_gpus"
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"
echo "train_data = $train_data"
echo "eval_data = $eval_data"
echo "model_name_or_path = $model_name_or_path"
echo "output_dir = $output_dir"
echo "gradient_accumulation_steps = $gradient_accumulation_steps"

# 设置默认值
: ${model_name_or_path:="Qwen/Qwen2.5-14B-Instruct"}
: ${output_dir:="../model_output/reranker_finetune_iter1_test_qwen_ep5_ds2"}
: ${num_train_epochs:=3}
: ${per_device_train_batch_size:=8}
: ${num_gpus:=2}
: ${train_data:="../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1.jsonl"}
: ${eval_data:="../data/embedder_train_eval_data/cross_validation/finetune_data_hn_from_emb_iter1_test.jsonl"}
: ${gradient_accumulation_steps:=4}

eval_retrieval_result_path="../model_output/icl_finetune_iter1_hn/retrieval_results_top25.jsonl"
eval_retrieval_sample_ratio=1.0

# set large epochs and small batch size for testing

use_qlora=False
train_group_size=16
deepspeed_config_path="./ds_stage2.json"

query_max_len=384
passage_max_len=64

learning_rate=2e-4
label_smoothing=0.0

save_merged_lora_model=False
save_steps=500

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path $model_name_or_path \
    --cache_dir $HF_HUB_CACHE \
    --use_lora True \
    --lora_rank 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --use_flash_attn True \
    --target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --save_merged_lora_model $save_merged_lora_model \
    --model_type decoder \
    --use_qlora $use_qlora \
"

data_args="\
    --train_data $train_data \
    --eval_data $eval_data \
    --eval_retrieval_result_path $eval_retrieval_result_path \
    --eval_retrieval_sample_ratio $eval_retrieval_sample_ratio \
    --cache_path ~/.cache \
    --train_group_size $train_group_size \
    --query_max_len $query_max_len \
    --passage_max_len $passage_max_len \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
    --query_instruction_for_rerank 'A: ' \
    --query_instruction_format '{}{}' \
    --passage_instruction_for_rerank 'B: ' \
    --passage_instruction_format '{}{}' \
"

training_args="\
    --output_dir $output_dir \
    --overwrite_output_dir \
    --learning_rate $learning_rate \
    --label_smoothing $label_smoothing \
    --save_lora_every_epoch True \
    --bf16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --dataloader_drop_last True \
    --warmup_ratio 0.05 \
    --gradient_checkpointing \
    --weight_decay 0.01 \
    --deepspeed $deepspeed_config_path \
    --logging_steps 1 \
    --save_steps $save_steps \
    --save_total_limit 5 \
"

cmd="torchrun --nproc_per_node $num_gpus \
    -m FlagEmbedding.finetune.reranker.decoder_only.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd