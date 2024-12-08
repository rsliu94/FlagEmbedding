import os
import torch
import logging
from typing import Optional, List
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer
from FlagEmbedding.utils.format_utils import get_detailed_example, get_detailed_instruct
from FlagEmbedding.utils.infer_utils import batch_to_device, get_new_queries, get_new_queries_examples_list, inference_doc, inference_query_examples_list
from FlagEmbedding.utils.data_utils import preprocess_text
from FlagEmbedding.utils.metrics import mean_average_precision_at_k, recall_at_k
import faiss
from transformers import TrainerCallback
import random
random.seed(42)

logger = logging.getLogger(__name__)


class SaveLoraCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        """每个epoch结束时被调用"""
        if not state.is_world_process_zero:
            return
        
        epoch = state.epoch
        output_dir = args.output_dir
        # 创建带有epoch编号的LoRA保存目录
        lora_output_dir = os.path.join(output_dir, f'lora_epoch_{int(epoch)}')
        os.makedirs(lora_output_dir, exist_ok=True)
        logger.info(f'Saving LoRA weights for epoch {int(epoch)} to {lora_output_dir}')
        
        if not hasattr(model.model, 'peft_config'):
            raise ValueError("模型不是PEFT模型，无法保存LoRA权重")
        
        try:
            # 保存LoRA权重和配置
            model.model.save_pretrained(
                lora_output_dir,
                save_embedding_layers="auto",
            )
            
            # 保存tokenizer配置
            if tokenizer is not None and state.is_world_process_zero:
                tokenizer.save_pretrained(lora_output_dir)
            
            # 保存训练参数
            if state.is_world_process_zero:
                torch.save(args, os.path.join(lora_output_dir, "training_args.bin"))
            
            logger.info("Successfully saved LoRA weights")
            
        except Exception as e:
            logger.error(f"Error saving LoRA weights: {str(e)}")
            raise


class DecoderOnlyEmbedderICLTrainer(AbsEmbedderTrainer):
    """
    Trainer class for base encoder models.
    """
    def __init__(self, eval_corpus_path, eval_queries_path, eval_examples_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_corpus_path = eval_corpus_path
        self.eval_queries_path = eval_queries_path
        self.eval_examples_path = eval_examples_path
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """Save the model to directory.

        Args:
            output_dir (Optional[str], optional): Output directory to save the model. Defaults to ``None``.

        Raises:
            NotImplementedError
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)

        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        # save the checkpoint for sentence-transformers library
        # if self.is_world_process_zero():
        #     save_ckpt_for_sentence_transformers(output_dir,
        #                                         pooling_mode=self.args.sentence_pooling_method,
        #                                         normlized=self.args.normlized)

    def save_lora_weights(self, output_dir: str = None):
        """只保存LoRA权重
    
        Args:
            output_dir (Optional[str], optional): 输出目录. 默认为 None，使用trainer的输出目录.
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if not hasattr(self.model.model, 'peft_config'):
            raise ValueError("模型不是PEFT模型，无法保存LoRA权重")
        
        try:
            # 保存LoRA权重和配置
            self.model.model.save_pretrained(
                output_dir,
                save_embedding_layers="auto",
            )
            
            # 保存tokenizer配置
            if self.tokenizer is not None and self.is_world_process_zero():
                self.tokenizer.save_pretrained(output_dir)
            
            # 保存训练参数
            if self.is_world_process_zero():
                torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            
            logger.info("Successfully saved LoRA weights")
            
        except Exception as e:
            logger.error(f"Error saving LoRA weights: {str(e)}")
            raise
        
    @torch.no_grad()
    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None):
        # debug: eval_llm_embedder.py results: 20s on doc + 2min on query; map@25_score: 0.20892443160013727 recall@25_score: 0.5653804930332261
        # evaluate(): [bs=16]26s on doc + 2:48 on query; map@25_score: 0.20677747849712655 recall@25_score: 0.5643086816720257
        # [if remove construct_name from task_description, map@25=0.188]
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        self.model.eval()
        eval_loss = None
        
        if eval_dataset is None and self.eval_dataset is not None:
            eval_dataset = self.eval_dataset
        
        if eval_dataset is not None:
            logger.info("Got eval dataset, calculating eval_loss on it ...")
            eval_loss = self._calculate_eval_loss(eval_dataset)
            logger.info(f"Eval loss: {eval_loss}")
            logger.info(f"Eval epoch done, refresh epoch")
            self.eval_dataset.refresh_epoch()
        
        if not self.is_world_process_zero():
            return {}
        
        if self.eval_corpus_path is None:
            logger.warning("No evaluation dataset provided. Skipping evaluation.")
            return
        
        logger.info("Evaluating the model for MAP@25 and Recall@25 metrics...")
        logger.info(f"DEBUG: corpus path: {self.eval_corpus_path}")
        logger.info(f"DEBUG: queries path: {self.eval_queries_path}")
        
        if self.eval_examples_path is not None:
            logger.info(f"DEBUG: examples path: {self.eval_examples_path}, loading examples dict [subject_id -> construct_id -> [instruct, query, response]]")
            with open(self.eval_examples_path, 'r') as f:
                examples_dict = json.load(f)
            logger.info(f"DEBUG: examples dict length: {len(examples_dict)}")
        
        corpus = [json.loads(line)['text'] for line in open(self.eval_corpus_path, 'r')] # list of strings
        print(f"Number of corpus: {len(corpus)}")
        correct_ids = [json.loads(line)['correct_id'] for line in open(self.eval_queries_path, 'r')] # list of floats
        print(f"Number of correct ids: {len(correct_ids)}")
        queries = []
        num_examples = 1
        use_examples_in_query = True
        print(f"Number of examples per query: {num_examples}")
        examples_prefix_list = []
        with open(self.eval_queries_path, 'r') as f:
            for line in f:
                row = json.loads(line)
                queries.append(get_detailed_instruct(task_description=row['prompt'], query=row['query']))
                subject_id = str(row['subject_id'])
                construct_id = str(row['construct_id'])
                if use_examples_in_query:
                    if subject_id in examples_dict:
                        if construct_id in examples_dict[subject_id]:
                            examples = []
                            N = len(examples_dict[subject_id][construct_id])
                            random_ids = random.sample(list(range(N)), min(N, num_examples))
                            for random_id in random_ids:
                                examples.append(get_detailed_example(examples_dict[subject_id][construct_id][random_id]['instruct'], 
                                                                examples_dict[subject_id][construct_id][random_id]['query'], 
                                                                examples_dict[subject_id][construct_id][random_id]['response']))
                            examples_prefix_list.append('\n\n'.join(examples) + '\n\n')
                        else:
                            random_construct_id = random.choice(list(examples_dict[subject_id].keys()))
                            examples = []
                            N = len(examples_dict[subject_id][random_construct_id])
                            random_ids = random.sample(list(range(N)), min(N, num_examples))
                            for random_id in random_ids:
                                examples.append(get_detailed_example(examples_dict[subject_id][random_construct_id][random_id]['instruct'], 
                                                            examples_dict[subject_id][random_construct_id][random_id]['query'], 
                                                            examples_dict[subject_id][random_construct_id][random_id]['response']))
                            examples_prefix_list.append('\n\n'.join(examples) + '\n\n')
                    else:
                        # random sample one
                        random_subject_id = random.choice(list(examples_dict.keys()))
                        random_construct_id = random.choice(list(examples_dict[random_subject_id].keys()))
                        N = len(examples_dict[random_subject_id][random_construct_id])
                        random_ids = random.sample(list(range(N)), min(N, num_examples))
                        for random_id in random_ids:
                            examples.append(get_detailed_example(examples_dict[random_subject_id][random_construct_id][random_id]['instruct'], 
                                                            examples_dict[random_subject_id][random_construct_id][random_id]['query'], 
                                                            examples_dict[random_subject_id][random_construct_id][random_id]['response']))
                        examples_prefix_list.append('\n\n'.join(examples) + '\n\n')
                else:
                    examples_prefix_list.append('')
        print(f"Number of examples prefix: {len(examples_prefix_list)}")
        print(f"Number of queries: {len(queries)}")
        
        query_max_len, doc_max_len = 512, 128
        batch_size = self.args.per_device_eval_batch_size
        device = next(self.model.parameters()).device
        logger.info(f"Eval batch size: {batch_size}")
        logger.info(f"Model device: {device}")
        logger.info("Check query/document token length...")
        cur_query_max_len = 0
        for query in tqdm(queries):
            cur_query_max_len = max(cur_query_max_len, len(self.tokenizer(query)['input_ids']))
        logger.info(f"Current query max length: {cur_query_max_len}")
        cur_doc_max_len = 0
        for doc in tqdm(corpus):
            cur_doc_max_len = max(cur_doc_max_len, len(self.tokenizer(doc)['input_ids']))
        logger.info(f"Current document max length: {cur_doc_max_len}")
        
        doc_embeddings = inference_doc(corpus, self.tokenizer, self.model, doc_max_len, batch_size, device)
        query_embeddings = inference_query_examples_list(queries, query_max_len, examples_prefix_list, self.tokenizer, self.model, batch_size, device)
        
        print("Building index...")
        index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        index.add(doc_embeddings)
        distances, indices = index.search(query_embeddings, k=100)
        print(f"Distances shape: {distances.shape}, Indices shape: {indices.shape}")

        for k in [25, 50, 75, 100]:
            print(f"--------------------------------")
            mapk_score = mean_average_precision_at_k(correct_ids, indices, k)
            print(f"map@{k}_score: {mapk_score}")
            recall_score = recall_at_k(correct_ids, indices, k)
            print(f"recall@{k}_score: {recall_score}")
            print(f"eval loss: {eval_loss}")
            print(f"--------------------------------")
        
        return {'map@25_score': mapk_score, 'recall@25_score': recall_score, 'eval_loss': eval_loss}

    def _calculate_eval_loss(self, eval_dataset: Dataset) -> float:
        """计算评估数据集上的损失

        Args:
            eval_dataset (Dataset): 评估数据集

        Returns:
            float: 评估损失
        """
        # 添加 DistributedSampler
        if self.args.local_rank != -1:  # 分布式训练时
            sampler = torch.utils.data.DistributedSampler(
                eval_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.local_rank,
                shuffle=False
            )
        else:
            sampler = None
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=1,
            collate_fn=self.data_collator,
            pin_memory=True,
            sampler=sampler  # 添加 sampler
        )
        
        total_loss = 0.0
        num_batches = 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = self._prepare_inputs(batch)
            loss = self.compute_loss(self.model, batch, compute_while_eval=True)
            total_loss += loss
            num_batches += 1
        
        # 在计算平均损失之前汇总所有GPU的结果
        if self.args.local_rank != -1:  # 分布式训练时
            # 汇总 total_loss
            total_loss_tensor = torch.tensor(total_loss).to(self.args.device)
            torch.distributed.all_reduce(total_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            total_loss = total_loss_tensor.item()
            
            # 汇总 num_batches
            num_batches_tensor = torch.tensor(num_batches).to(self.args.device)
            torch.distributed.all_reduce(num_batches_tensor, op=torch.distributed.ReduceOp.SUM)
            num_batches = num_batches_tensor.item()
        return total_loss / num_batches if num_batches > 0 else float('inf')