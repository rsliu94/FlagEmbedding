import logging
from typing import Tuple
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer

from FlagEmbedding.abc.finetune.embedder.AbsArguments import AbsEmbedderDataArguments, AbsEmbedderTrainingArguments
from FlagEmbedding.abc.finetune.embedder import AbsEmbedderRunner, AbsEmbedderModel, EmbedderTrainerCallbackForDataRefresh, EvaluateCallback, AbsEmbedderEvalDataset

from .arguments import DecoderOnlyEmbedderModelArguments, DecoderOnlyEmbedderDataArguments
from .trainer import DecoderOnlyEmbedderTrainer, SaveLoraCallback
from .modeling import BiDecoderOnlyEmbedderModel
from .load_model import get_model, save_merged_model

logger = logging.getLogger(__name__)


class DecoderOnlyEmbedderRunner(AbsEmbedderRunner):
    """Runner class for decoder only embedding model.

    Args:
        model_args (DecoderOnlyEmbedderModelArguments): Model arguments instance.
        data_args (AbsEmbedderDataArguments): Data arguments instance.
        training_args (AbsEmbedderTrainingArguments): Trainer arguments.
    """
    def __init__(
        self,
        model_args: DecoderOnlyEmbedderModelArguments,
        data_args: DecoderOnlyEmbedderDataArguments,
        training_args: AbsEmbedderTrainingArguments
    ):
        super().__init__(model_args, data_args, training_args)
        self.data_args = data_args

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

        num_labels = 1
        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=num_labels,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        logger.info('Config: %s', config)

        model = BiDecoderOnlyEmbedderModel(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings,
            config=config
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        if self.training_args.fix_position_embedding:
            for k, v in model.named_parameters():
                if "position_embeddings" in k:
                    logging.info(f"Freeze the parameters for {k}")
                    v.requires_grad = False
        return tokenizer, model

    def load_trainer(self) -> DecoderOnlyEmbedderTrainer:
        """Load the trainer.

        Returns:
            DecoderOnlyEmbedderTrainer: Loaded trainer instance.
        """
        trainer = DecoderOnlyEmbedderTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            eval_corpus_path=self.data_args.eval_corpus_path,
            eval_queries_path=self.data_args.eval_queries_path
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(EmbedderTrainerCallbackForDataRefresh(self.train_dataset))
        if self.data_args.eval_corpus_path is not None and self.data_args.eval_queries_path is not None:
            logger.info('Add EvaluateCallback')
            trainer.add_callback(EvaluateCallback())
        if self.training_args.save_lora_every_epoch:
            logger.info('Add SaveLoraCallback')
            trainer.add_callback(SaveLoraCallback())
        return trainer
    
    def load_eval_dataset(self) -> AbsEmbedderEvalDataset:
        """Load the evaluation dataset.

        Returns:
            DecoderOnlyEmbedderEvalDataset: The loaded dataset instance.
        """
        eval_dataset = AbsEmbedderEvalDataset(
                args=self.data_args,
                tokenizer=self.tokenizer
        )
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
        self.trainer.save_model()

        # save merged model
        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            save_merged_model(self.model_args, self.training_args.output_dir)
