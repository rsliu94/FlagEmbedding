import os
import torch
import logging
from typing import Optional, List
# from transformers.deepspeed import is_deepspeed_zero3_enabled
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from FlagEmbedding.abc.finetune.reranker import AbsRerankerTrainer
import json
from FlagEmbedding.utils.infer_utils import get_inputs
from FlagEmbedding.utils.metrics import mean_average_precision_at_k, recall_at_k
from FlagEmbedding.utils.constants import RERANKER_PROMPT
import numpy as np
from torch import Tensor
logger = logging.getLogger(__name__)


def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

class DecoderOnlyRerankerTrainer(AbsRerankerTrainer):
    """
    Trainer class for encoder only base reranker models.
    """
    def __init__(self, eval_retrieval_result_path, eval_retrieval_sample_ratio, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_retrieval_result_path = eval_retrieval_result_path
        self.eval_retrieval_sample_ratio = eval_retrieval_sample_ratio
        
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

        # if is_deepspeed_zero3_enabled():
        #     if state_dict is None:
        #         state_dict = self.model.state_dict()
        #     prefix = 'model.'
        #     assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        #     state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        #     lora_state_dict = get_peft_model_state_dict(self.model.model, state_dict)
        #     if self.args.process_index <= 0:
        #         torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
        #         print(f"Save adapter model at {output_dir}")

    @torch.no_grad()
    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None):
        self._memory_tracker.start()
        self.model.eval()
        eval_loss = None
        
        if eval_dataset is None and self.eval_dataset is not None:
            eval_dataset = self.eval_dataset
            
        if eval_dataset is not None:
            logger.info("Got eval dataset, calculating eval_loss on it ...")
            eval_loss = self._calculate_eval_loss(eval_dataset)
            logger.info(f"Eval loss: {eval_loss}")
        
        if not self.is_world_process_zero():
            return {}
        
        if self.eval_retrieval_result_path is None:
            logger.warning("No evaluation retrieval result provided. Skipping evaluation.")
            return
        
        logger.info("Evaluating the model for MAP@25 and Recall@25 metrics...")
        logger.info(f"DEBUG: retrieval result path: {self.eval_retrieval_result_path}")
        
        retrievals = [json.loads(line) for line in open(self.eval_retrieval_result_path, 'r')]
        if self.eval_retrieval_sample_ratio < 1.0:
            logger.info(f"DEBUG: eval retrieval sample ratio: {self.eval_retrieval_sample_ratio}")
            logger.info(f"original retrieval length: {len(retrievals)}")
            retrievals = retrievals[:int(len(retrievals) * self.eval_retrieval_sample_ratio)]
            logger.info(f"sampled retrieval length: {len(retrievals)}")
            
        pairs = []
        for retrieval in retrievals:
            query = retrieval['query']
            candidates = retrieval['candidate_texts']
            pairs.extend( [[query, candidate] for candidate in candidates])
        
        batch_size = 16
        scores = []
        for i in tqdm(range(0, len(pairs), batch_size), desc="Evaluating Metrics"):
            batch_pairs = pairs[i:i+batch_size]
            batch_inputs = get_inputs(batch_pairs, prompt=RERANKER_PROMPT, tokenizer=self.tokenizer, max_length=512)
            batch_inputs = batch_to_device(batch_inputs, next(self.model.parameters()).device)
            scores_tensor = self.model.encode(batch_inputs).view(-1, ).float()
            scores.extend(scores_tensor.tolist())
        logger.info(f"DEBUG: scores length: {len(scores)}")

        # 处理每个检索结果的重排序
        score_idx = 0
        for retrieval in retrievals:
            num_candidates = len(retrieval['candidate_texts'])
            # 获取当前查询的所有候选文档得分
            current_scores = scores[score_idx:score_idx + num_candidates]
            # 将候选ID和得分配对并排序
            id_score_pairs = list(zip(retrieval['candidate_ids'], current_scores))
            sorted_pairs = sorted(id_score_pairs, key=lambda x: x[1], reverse=True)
            # 提取排序后的ID
            retrieval['reranked_ids'] = [pair[0] for pair in sorted_pairs]
            score_idx += num_candidates
        
        # 准备评估数据
        correct_ids = [retrieval['correct_id'] for retrieval in retrievals]
        reranked_ids = [retrieval['reranked_ids'] for retrieval in retrievals]
        recall_ids = [retrieval['candidate_ids'] for retrieval in retrievals]

        # 计算评估指标
        logger.info("Calculating MAP@25 and Recall@25 metrics... with sample ratio: {self.eval_retrieval_sample_ratio}")
        print("==Rerank==")
        mapk_score = mean_average_precision_at_k(correct_ids, np.array(reranked_ids), 25)
        print(f"map@25_score: {mapk_score}")

        recall_score = recall_at_k(correct_ids, np.array(reranked_ids), 25)
        print(f"recall@25_score: {recall_score}")

        print("==Recall==")
        mapk_score = mean_average_precision_at_k(correct_ids, np.array(recall_ids), 25)
        print(f"map@25_score: {mapk_score}")

        recall_score = recall_at_k(correct_ids, np.array(recall_ids), 25)
        print(f"recall@25_score: {recall_score}")
    
        
        return {}
    
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
        