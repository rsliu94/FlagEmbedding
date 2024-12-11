from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig
import torch
import os
from peft import PeftModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_model_path', type=str, default='BAAI/bge-en-icl')
parser.add_argument('--type', type=str, default='reranker', choices=['reranker', 'embedder'])
parser.add_argument('--lora_path', type=str, default='../model_output/icl_finetune_round1/lora_epoch_1')
parser.add_argument('--output_dir', type=str, default='../model_output/icl_finetune_round1/merged_model_lora_epoch_1')
args = parser.parse_args()

# show args
print("\nScript arguments:")
for arg, value in vars(args).items():
    print(f"  {arg}: {value}")
print()

print("Loading config from {}".format(args.base_model_path))
config = AutoConfig.from_pretrained(args.base_model_path)

if args.type == 'embedder':
    print("Loading standard model AutoModel from {}".format(args.base_model_path))
    model = AutoModel.from_pretrained(args.base_model_path, config=config)
elif args.type == 'reranker':
    print("Loading standard model AutoModelForCausalLM from {}".format(args.base_model_path))
    model = AutoModelForCausalLM.from_pretrained(args.base_model_path, config=config)

print("Loading lora model from {}".format(args.lora_path))
model = PeftModel.from_pretrained(model, args.lora_path)
print("Merging lora model with base model")
model = model.merge_and_unload()

print("Saving merged model to {}".format(args.output_dir))
model.save_pretrained(args.output_dir)

print("Loading tokenizer from {}".format(args.lora_path))
tokenizer = AutoTokenizer.from_pretrained(args.lora_path)
print("Saving tokenizer to {}".format(args.output_dir))
tokenizer.save_pretrained(args.output_dir)
print("Done")
