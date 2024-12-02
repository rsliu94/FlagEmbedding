import os
import torch
import logging
from typing import Optional, List
import json
from tqdm import tqdm
import numpy as np
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer
from FlagEmbedding.utils.format_utils import get_detailed_example, get_detailed_instruct
from FlagEmbedding.utils.infer_utils import batch_to_device, get_new_queries
from FlagEmbedding.utils.data_utils import preprocess_text
from FlagEmbedding.utils.metrics import mean_average_precision_at_k, recall_at_k
import faiss

logger = logging.getLogger(__name__)


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
        self.model.model.save_pretrained(output_dir)
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
    def evaluate(self, ignore_keys: Optional[List[str]] = None):
        # debug: eval_llm_embedder.py results: 20s on doc + 2min on query; map@25_score: 0.20892443160013727 recall@25_score: 0.5653804930332261
        # evaluate(): [bs=16]26s on doc + 2:48 on query; map@25_score: 0.20677747849712655 recall@25_score: 0.5643086816720257
        # [if remove construct_name from task_description, map@25=0.188]
        # memory metrics - must set up as early as possible 
        if not self.is_world_process_zero():
            return {}
        
        self._memory_tracker.start()
        self.model.eval()
        if self.eval_corpus_path is None:
            logger.warning("No evaluation dataset provided. Skipping evaluation.")
            return
        
        logger.info("Evaluating the model...")
        logger.info(f"DEBUG: corpus path: {self.eval_corpus_path}")
        logger.info(f"DEBUG: queries path: {self.eval_queries_path}")
        if self.eval_examples_path is not None:
            logger.info(f"DEBUG: examples path: {self.eval_examples_path}")
            with open(self.eval_examples_path, 'r', encoding='utf-8') as f:
                examples = json.load(f)
            examples = [get_detailed_example(e['instruct'], e['query'], e['response']) for e in examples]
            examples_prefix = '\n\n'.join(examples) + '\n\n' # if there not exists any examples, just set examples_prefix = ''
        else:
            logger.info("No evaluation examples provided.")
            examples_prefix = ''
        logger.info(f"Use examples_prefix: {examples_prefix}")
        
        corpus = [json.loads(line)['text'] for line in open(self.eval_corpus_path, 'r')] # list of strings
        print(f"Number of corpus: {len(corpus)}")
        correct_ids = [json.loads(line)['correct_id'] for line in open(self.eval_queries_path, 'r')] # list of floats
        print(f"Number of correct ids: {len(correct_ids)}")
        queries = []
        with open(self.eval_queries_path, 'r') as f:
            for line in f:
                row = json.loads(line)
                task_description = f'Given a math question about {preprocess_text(row["construct_name"])} and a misconcepted incorrect answer to it, retrieve the most accurate reason for the misconception leading to the incorrect answer.'
                # task_description = 'Given a math question and a misconcepted incorrect answer to it, retrieve the most accurate reason for the misconception leading to the incorrect answer.'
                query = f'{row["question"]} \n Incorrect answer : {row["wrong_answer"]}'
                queries.append(get_detailed_instruct(task_description=task_description, query=query))
        print(f"Number of queries: {len(queries)}")
        
        query_max_len, doc_max_len = 384, 128
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
        
        doc_embeddings = []
        print("Getting document embeddings...")
        for i in tqdm(range(0, len(corpus), batch_size)):
            batch = corpus[i:i+batch_size]
            batch_dict = self.tokenizer(batch, max_length=doc_max_len, padding=True, truncation=True, return_tensors='pt')
            batch_dict = batch_to_device(batch_dict, device)
            embedding = self.model.encode(batch_dict)
            embedding = embedding.detach().cpu().numpy()
            doc_embeddings.append(embedding)
        doc_embeddings = np.concatenate(doc_embeddings, axis=0)
        print(f"Document embeddings shape: {doc_embeddings.shape}")
        
        print("Getting query embeddings...")
        query_embeddings = []
        for i in tqdm(range(0, len(queries), batch_size)):
            batch = queries[i:i+batch_size]
            new_max_length, new_queries = get_new_queries(batch, query_max_len, examples_prefix, self.tokenizer)
            batch_dict = self.tokenizer(new_queries, max_length=new_max_length, padding=True, truncation=True, return_tensors='pt')
            batch_dict = batch_to_device(batch_dict, device)
            embedding = self.model.encode(batch_dict)
            embedding = embedding.detach().cpu().numpy()
            query_embeddings.append(embedding)
        query_embeddings = np.concatenate(query_embeddings, axis=0)
        print(f"Query embeddings shape: {query_embeddings.shape}")
        
        print("Building index...")
        index = faiss.IndexFlatL2(doc_embeddings.shape[1])
        index.add(doc_embeddings)
        distances, indices = index.search(query_embeddings, k=25)
        print(f"Distances shape: {distances.shape}, Indices shape: {indices.shape}")

        mapk_score = mean_average_precision_at_k(correct_ids, indices, 25)
        print(f"map@25_score: {mapk_score}")

        recall_score = recall_at_k(correct_ids, indices, 25)
        print(f"recall@25_score: {recall_score}")
        
        return {'map@25_score': mapk_score, 'recall@25_score': recall_score}
