import logging
from typing import Tuple
import os
import torch
from pathlib import Path
from FlagEmbedding.abc.finetune.reranker.AbsArguments import AbsRerankerDataArguments, AbsRerankerTrainingArguments
from transformers import (
    AutoTokenizer, PreTrainedTokenizer
)
from transformers.trainer import TrainerCallback
from FlagEmbedding.abc.finetune.reranker import AbsRerankerRunner, AbsRerankerModel
from FlagEmbedding.abc.finetune.reranker.AbsDataset import AbsLLMRerankerEvalDataset

from .modeling import CrossDecoderModel
from .arguments import RerankerModelArguments
from .trainer import DecoderOnlyRerankerTrainer
from .load_model import get_model, save_merged_model

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


class EvaluateCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Callback method triggered at the end of each epoch.
        Performs model evaluation.
        
        Args:
            args: Training arguments
            state: Current training state
            control: Training control object
            kwargs: Additional keyword arguments (includes trainer and model)
        """
        # Ensure evaluation happens after each epoch
        control.should_evaluate = True
        
        # Optional: Add custom logging or additional actions
        print(f"Epoch {state.epoch} completed. Running evaluation...")
        
        return control


class DecoderOnlyRerankerRunner(AbsRerankerRunner):
    """
    Decoder only reranker runner for finetuning.
    
    Args:
        model_args (RerankerModelArguments): Model arguments instance.
        data_args (AbsRerankerDataArguments): Data arguments instance.
        training_args (AbsRerankerTrainingArguments): Trainer arguments.
    """
    def __init__(
        self,
        model_args: RerankerModelArguments,
        data_args: AbsRerankerDataArguments,
        training_args: AbsRerankerTrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)

    def load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, AbsRerankerModel]:
        """Load the tokenizer and model.

        Returns:
            Tuple[PreTrainedTokenizer, AbsEmbedderModel]: Tokenizer and model instances.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            token=self.model_args.token,
            cache_dir=self.model_args.cache_dir,
            use_fast=False,
            add_eos_token=False,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        if tokenizer.pad_token is None:
            if tokenizer.unk_token is not None:
                tokenizer.pad_token = tokenizer.unk_token
                tokenizer.pad_token_id = tokenizer.unk_token_id
            elif tokenizer.eod_id is not None:
                tokenizer.pad_token = tokenizer.eod
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.bos_token = tokenizer.im_start
                tokenizer.bos_token_id = tokenizer.im_start_id
                tokenizer.eos_token = tokenizer.im_end
                tokenizer.eos_token_id = tokenizer.im_end_id
            else:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
        # if 'mistral' in self.model_args.model_name_or_path.lower():
        tokenizer.padding_side = 'left'

        base_model = get_model(self.model_args)

        model = CrossDecoderModel(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=self.training_args.per_device_train_batch_size,
            label_smoothing=self.model_args.label_smoothing,
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        return tokenizer, model

    def load_trainer(self) -> DecoderOnlyRerankerTrainer:
        """Load the trainer.

        Returns:
            DecoderOnlyRerankerTrainer: Loaded trainer instance.
        """
        trainer = DecoderOnlyRerankerTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            eval_retrieval_result_path=self.data_args.eval_retrieval_result_path,
            eval_retrieval_sample_ratio=self.data_args.eval_retrieval_sample_ratio,
        )
        if self.data_args.eval_data is not None:
            logger.info('Add EvaluateCallback')
            trainer.add_callback(EvaluateCallback())
        if self.training_args.save_lora_every_epoch:
            logger.info('Add SaveLoraCallback')
            trainer.add_callback(SaveLoraCallback())
        return trainer

    def run(self):
        """
        Run the finetuning.
        """
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)

        # Training
        if self.training_args.resume_from_checkpoint and self.training_args.resume_from_checkpoint == 'True':
            self.training_args.resume_from_checkpoint = True
        logger.info(f'Resume from checkpoint: {self.training_args.resume_from_checkpoint}, type: {type(self.training_args.resume_from_checkpoint)}')
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()

        # save merged model
        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            save_merged_model(self.model_args, self.training_args.output_dir)
