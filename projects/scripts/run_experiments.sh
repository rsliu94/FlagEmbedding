#!/bin/bash

# 定义参数组合
declare -a params=(
    "16 32 1e-4"
    "32 16 1e-4"
    "32 32 1e-4"
    "32 64 1e-4"
    "64 32 1e-4"
    "64 64 1e-4"
    "64 128 1e-4"
    "128 64 1e-4"
)

# 创建日志目录
log_dir="./lora_experiments/logs"
mkdir -p $log_dir

# 循环执行每组参数
for param in "${params[@]}"; do
    read -r rank alpha lr <<< "$param"
    echo "Running with rank=$rank, alpha=$alpha, lr=$lr"
    
    # 创建带时间戳的日志文件名
    timestamp=$(date +"%Y%m%d_%H%M%S")
    log_file="$log_dir/lora_rank${rank}_alpha${alpha}_lr${lr}_${timestamp}.log"
    
    # 执行训练脚本并将输出重定向到日志文件
    ./icl_finetune.sh $rank $alpha $lr 2>&1 | tee $log_file
    
    echo "Experiment completed. Log saved to: $log_file"
    echo "----------------------------------------"
done 

# Usage:
# chmod +x run_experiments.sh
# ./run_experiments.sh ; /usr/bin/shutdown  