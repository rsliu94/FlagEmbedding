import os
import torch
import logging
from typing import Optional, List
from torch.utils.data import DataLoader, Dataset
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer
import json
import random
from tqdm import tqdm
from FlagEmbedding.utils.format_utils import get_detailed_example, get_detailed_instruct
from FlagEmbedding.utils.infer_utils import inference_doc, inference_query_base
from FlagEmbedding.utils.metrics import mean_average_precision_at_k, recall_at_k
import faiss
from peft import get_peft_model_state_dict
random.seed(42)

logger = logging.getLogger(__name__)

class DecoderOnlyEmbedderTrainer(AbsEmbedderTrainer):
    """
    Trainer class for base encoder models.
    """
    def __init__(self, eval_corpus_path, eval_queries_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_corpus_path = eval_corpus_path
        self.eval_queries_path = eval_queries_path
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
        
        if state_dict is None:
            state_dict = self.model.state_dict()
        prefix = 'model.'
        assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        lora_state_dict = get_peft_model_state_dict(self.model.model, state_dict)
        if self.args.process_index <= 0:
            torch.save(lora_state_dict, os.path.join(output_dir, "adapter_model.bin"))
            print(f"Save adapter model at {output_dir}")
        
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
        
        if not self.is_world_process_zero():
            return {}
        
        if self.eval_corpus_path is None or self.eval_queries_path is None:
            logger.warning("No evaluation dataset provided. Skipping evaluation.")
            return
        
        logger.info("Evaluating the model for MAP@25 and Recall@25 metrics...")
        logger.info(f"DEBUG: corpus path: {self.eval_corpus_path}")
        logger.info(f"DEBUG: queries path: {self.eval_queries_path}")
        
        corpus = [json.loads(line)['text'] for line in open(self.eval_corpus_path, 'r')] # list of strings
        print(f"Number of corpus: {len(corpus)}")
        correct_ids = [json.loads(line)['correct_id'] for line in open(self.eval_queries_path, 'r')] # list of floats
        print(f"Number of correct ids: {len(correct_ids)}")
        queries = []


        with open(self.eval_queries_path, 'r') as f:
            for line in f:
                row = json.loads(line)
                queries.append(get_detailed_instruct(task_description=row['prompt'], query=row['query']))

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
        query_embeddings = inference_query_base(queries, self.tokenizer, self.model, query_max_len, batch_size, device)
        
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