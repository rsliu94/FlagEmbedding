# sh icl_finetune.sh 2>&1 | tee ./logs/icl_hn_finetune_round2_$(date +%Y%m%d_%H%M%S).log
# sh icl_finetune.sh --epochs 1 --batch_size 8 --num_gpus 2 --gpu_ids "0,1" 2>&1 | tee ./logs/icl_finetune_iter0_hn_$(date +%Y%m%d_%H%M%S).log
# sh embedder_finetune.sh --epochs 1 --batch_size 1 --num_gpus 1 --gpu_ids "4" 2>&1 | tee ./logs/emb_qwen_finetune_iter0_hn_$(date +%Y%m%d_%H%M%S).log
# export WANDB_MODE=disabled
# echo 'export HF_TOKEN=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' >> ~/.bashrc
# 科学上网
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
    *)
      echo "错误: 未知参数 '$key'"
      echo "可用参数: --epochs, --batch_size, --num_gpus, --gpu_ids"
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

# 设置默认值
: ${num_train_epochs:=3}
: ${per_device_train_batch_size:=1}
: ${num_gpus:=1}

train_data="\
    ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn.jsonl \
"
eval_data="\
    ../data/embedder_train_eval_data/cross_validation/finetune_data_iter0_hn_test.jsonl  \
"
eval_corpus_path="../data/embedder_train_eval_data/cross_validation/corpus.jsonl"
eval_queries_path="../data/embedder_train_eval_data/cross_validation/test_queries.jsonl"

query_max_len=128
passage_max_len=64
gradient_accumulation_steps=1

save_merged_lora_model=True
save_steps=219
output_dir="../model_output/emb_qwen_finetune_iter0_hn"

lora_rank=8
lora_alpha=16
learning_rate=1e-4

model_name_or_path=Qwen/Qwen2.5-14B-Instruct

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

model_args="\
    --model_name_or_path $model_name_or_path \
    --cache_dir $HF_HUB_CACHE \
    --use_lora True \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --use_qlora True \
    --target_modules q_proj k_proj v_proj o_proj gate_proj down_proj up_proj \
    --additional_special_tokens '<instruct>' '<query>' \
    --save_merged_lora_model $save_merged_lora_model \
"

training_args="\
    --save_lora_every_epoch True \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --learning_rate $learning_rate \
    --fp16 \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_train_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    --gradient_checkpointing \
    --deepspeed ./ds_stage2_rerank.json \
    --logging_steps 1 \
    --save_total_limit 2 \
    --save_steps $save_steps \
    --negatives_cross_device \
    --temperature 0.02 \
    --sentence_pooling_method last_token \
    --normalize_embeddings True \
    --kd_loss_type m3_kd_loss \
"

data_args="\
    --train_data $train_data \
    --eval_data $eval_data \
    --eval_corpus_path $eval_corpus_path \
    --eval_queries_path $eval_queries_path \
    --cache_path ~/.cache \
    --train_group_size 8 \
    --query_max_len $query_max_len \
    --passage_max_len $passage_max_len \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval \"Given a multiple choice math question and a student's incorrect answer choice, identify and retrieve the specific mathematical misconception or error in the student's thinking that led to this wrong answer.\" \
    --query_instruction_format '<instruct>{}\n<query>{}' \
    --knowledge_distillation False \
"

MASTER_PORT=$(shuf -i 29502-39999 -n 1)
echo "使用端口: ${MASTER_PORT}"
cmd="torchrun --nproc_per_node $num_gpus \
    --master_port $MASTER_PORT \
    -m FlagEmbedding.finetune.embedder.decoder_only.base \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd