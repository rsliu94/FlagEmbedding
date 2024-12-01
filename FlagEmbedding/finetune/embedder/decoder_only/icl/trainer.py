import os
import torch
import logging
from typing import Optional

from FlagEmbedding.abc.finetune.embedder import AbsEmbedderTrainer

logger = logging.getLogger(__name__)


class DecoderOnlyEmbedderICLTrainer(AbsEmbedderTrainer):
    """
    Trainer class for base encoder models.
    """
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
