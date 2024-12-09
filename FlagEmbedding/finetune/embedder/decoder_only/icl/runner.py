import logging
from typing import Tuple
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, TrainerCallback

import os
from FlagEmbedding.abc.finetune.embedder.AbsArguments import AbsEmbedderTrainingArguments
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderRunner, AbsEmbedderModel, EmbedderTrainerCallbackForDataRefresh, EvaluateCallback

from .arguments import DecoderOnlyEmbedderICLModelArguments, DecoderOnlyEmbedderICLDataArguments
from .trainer import DecoderOnlyEmbedderICLTrainer
from .modeling import BiDecoderOnlyEmbedderICLModel
from .dataset import DecoderOnlyEmbedderICLSameDatasetTrainDataset, DecoderOnlyEmbedderICLSameDatasetEvalDataset
from .load_model import get_model, save_merged_model

logger = logging.getLogger(__name__)

class SaveLoraCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):

        """每个epoch结束时被调用"""
        
        control.should_save = True
        logger.info(f"DEBUG: should_save: {control.should_save}")
        logger.info(f"Epoch {state.epoch} completed. Saving LoRA model...")
        return control
    
        # if not state.is_world_process_zero:
        #     return
        
        # epoch = state.epoch
        # output_dir = args.output_dir
        # # 创建带有epoch编号的LoRA保存目录
        # lora_output_dir = os.path.join(output_dir, f'lora_epoch_{int(epoch)}')
        # os.makedirs(lora_output_dir, exist_ok=True)
        # logger.info(f'Saving LoRA weights for epoch {int(epoch)} to {lora_output_dir}')
        
        # if not hasattr(model.model, 'peft_config'):
        #     raise ValueError("模型不是PEFT模型，无法保存LoRA权重")
        
        # try:
        #     # 保存LoRA权重和配置
        #     model.model.save_pretrained(
        #         lora_output_dir,
        #         save_embedding_layers="auto",
        #     )
            
        #     # 保存tokenizer配置
        #     if tokenizer is not None and state.is_world_process_zero:
        #         tokenizer.save_pretrained(lora_output_dir)
            
        #     # 保存训练参数
        #     if state.is_world_process_zero:
        #         torch.save(args, os.path.join(lora_output_dir, "training_args.bin"))
            
        #     logger.info("Successfully saved LoRA weights")
            
        # except Exception as e:
        #     logger.error(f"Error saving LoRA weights: {str(e)}")
        #     raise

class DecoderOnlyEmbedderICLRunner(AbsEmbedderRunner):
    """Runner class for decoder only icl model.

    Args:
        model_args (DecoderOnlyEmbedderICLModelArguments): Model arguments instance.
        data_args (DecoderOnlyEmbedderICLDataArguments): Data arguments instance.
        training_args (AbsEmbedderTrainingArguments): Trainer arguments.
    """
    def __init__(
        self,
        model_args: DecoderOnlyEmbedderICLModelArguments,
        data_args: DecoderOnlyEmbedderICLDataArguments,
        training_args: AbsEmbedderTrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)
        self.model_args: DecoderOnlyEmbedderICLModelArguments
        self.data_args: DecoderOnlyEmbedderICLDataArguments
        self.training_args: AbsEmbedderTrainingArguments

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsEmbedderModel]:
        """Load tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            token=self.model_args.token,
            cache_dir=self.model_args.cache_dir,
            use_fast=False,
            add_eos_token=True
        )

        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'

        resize = False
        if self.model_args.additional_special_tokens is not None:
            special_tokens_dict = {'additional_special_tokens': self.model_args.additional_special_tokens}
            add_num = tokenizer.add_special_tokens(special_tokens_dict)
            if add_num > 0:
                resize = True
                logger.info(f"Add {add_num} special tokens to the tokenizer. Special tokens: {self.model_args.additional_special_tokens}")
            else:
                logger.warning(f"Special tokens {self.model_args.additional_special_tokens} already exists in the tokenizer.")
        base_model = get_model(self.model_args, self.training_args.output_dir, resize, len(tokenizer))
        # base_model: transformers.PreTrainedModel or PeftModel, The loaded model.

        num_labels = 1
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        logger.info('Config: %s', config)

        model = BiDecoderOnlyEmbedderICLModel(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings
        ) # model.model is PeftModel
        # PeftModel.save_pretrained("output_dir") only save the extra PEFT weights that were trained, meaning it is super efficient to store

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False
        return tokenizer, model

    def load_trainer(self) -> DecoderOnlyEmbedderICLTrainer:
        """Load the trainer.

        Returns:
            DecoderOnlyEmbedderICLTrainer: Loaded trainer instance.
        """
        trainer = DecoderOnlyEmbedderICLTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            eval_corpus_path=self.data_args.eval_corpus_path,
            eval_queries_path=self.data_args.eval_queries_path,
            eval_examples_path=self.data_args.eval_examples_path
        )
        # 添加以下调试代码
        # logger.info("=== Trainer Attributes ===")
        # for attr in dir(trainer):
        #     if not attr.startswith('_'):  # 跳过私有属性
        #         try:
        #             value = getattr(trainer, attr)
        #             logger.info(f"{attr}: {value}")
        #         except Exception as e:
        #             logger.info(f"{attr}: <无法获取值: {str(e)}>")
        # logger.info("=====================")
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(EmbedderTrainerCallbackForDataRefresh(self.train_dataset))
        if self.data_args.eval_corpus_path is not None and self.data_args.eval_queries_path is not None:
            logger.info('Add EvaluateCallback')
            trainer.add_callback(EvaluateCallback())
        if self.training_args.save_lora_every_epoch:
            logger.info('Add SaveLoraCallback')
            trainer.add_callback(SaveLoraCallback())
        return trainer

    def load_train_dataset(self) -> DecoderOnlyEmbedderICLSameDatasetTrainDataset:
        """Load the dataset instance for training.

        Raises:
            NotImplementedError: Only support `same_dataset_within_batch` for `DecoderOnlyEmbedderICLRunner`.

        Returns:
            DecoderOnlyEmbedderICLSameDatasetTrainDataset: The dataset instance.
        """
        if self.data_args.same_dataset_within_batch:
            train_dataset = DecoderOnlyEmbedderICLSameDatasetTrainDataset(
                args=self.data_args,
                default_batch_size=self.training_args.per_device_train_batch_size,
                seed=self.training_args.seed,
                tokenizer=self.tokenizer,
                process_index=self.training_args.process_index,
                num_processes=self.training_args.world_size
            )
            self.training_args.per_device_train_batch_size = 1
            self.training_args.dataloader_num_workers = 0   # avoid multi-processing
        else:
            raise NotImplementedError("Only support `same_dataset_within_batch` for `DecoderOnlyEmbedderICLRunner`.")
        return train_dataset
    
    def load_eval_dataset(self) -> DecoderOnlyEmbedderICLSameDatasetEvalDataset:
        """Load the dataset instance for evaluation.

        Returns:
            DecoderOnlyEmbedderICLSameDatasetEvalDataset: The evaluation dataset instance.
        """
        if self.data_args.same_dataset_within_batch:
            if self.data_args.eval_data:
                eval_dataset = DecoderOnlyEmbedderICLSameDatasetEvalDataset(
                    args=self.data_args,
                    default_batch_size=self.training_args.per_device_eval_batch_size,
                    seed=self.training_args.seed,
                    tokenizer=self.tokenizer,
                    process_index=self.training_args.process_index,
                    num_processes=self.training_args.world_size
                )
            else:
                return None
        else:
            raise NotImplementedError("Only support `same_dataset_within_batch` for `DecoderOnlyEmbedderICLRunner`.")
        return eval_dataset

    def run(self):
        """
        Run the finetune.
        """
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Training
        if self.training_args.resume_from_checkpoint and self.training_args.resume_from_checkpoint == 'True':
            self.training_args.resume_from_checkpoint = True
        logger.info(f'Resume from checkpoint: {self.training_args.resume_from_checkpoint}, type: {type(self.training_args.resume_from_checkpoint)}')
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        # self.trainer.evaluate(self.eval_dataset)
        
        # save merged model
        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            logger.info('Saving merged model')
            self.trainer.save_model()
            save_merged_model(self.model_args, self.training_args.output_dir)
