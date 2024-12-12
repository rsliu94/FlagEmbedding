# ### usage
# sh projects/submit/scp_lora_to_local.sh projects/model_output/cross_validation/reranker_finetune_qwen14b_iter1_with_kd/lora_epoch_3rd projects/model_output/cross_validation/reranker_finetune_qwen14b_iter1_with_kd

### scp models to local machine
path_to_lora=$1
target_path=$2

# 检查是否提供了参数
if [ -z "$path_to_lora" ]; then
    echo "错误: 请提供 LoRA 路径参数"
    exit 1
fi

# 确保本地目标目录存在
local_dir="/Users/runshengliu/github/FlagEmbedding"
mkdir -p "${local_dir}/${target_path}"

# 执行 scp 命令并检查结果
scp -rP 36485 "root@connect.bjc1.seetacloud.com:/root/autodl-tmp/github/FlagEmbedding/$path_to_lora" "${local_dir}/$target_path" || {
    echo "错误: SCP 传输失败"
    exit 1
}

echo "传输完成"