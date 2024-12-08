import torch
from transformers import PreTrainedModel, AutoTokenizer
import logging
from torch import nn
from FlagEmbedding.abc.finetune.reranker import AbsRerankerModel

logger = logging.getLogger(__name__)


class CrossDecoderModel(AbsRerankerModel):
    """
    Model class for decoder only reranker.

    Args:
        base_model (PreTrainedModel): The underlying pre-trained model used for encoding and scoring input pairs.
        tokenizer (AutoTokenizer, optional): The tokenizer for encoding input text. Defaults to ``None``.
        train_batch_size (int, optional): The batch size to use. Defaults to ``4``.
    """
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: AutoTokenizer = None,
        train_batch_size: int = 4,
        label_smoothing: float = 0.0,
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            train_batch_size=train_batch_size,
        )
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean', label_smoothing=label_smoothing)
        
    def encode(self, features):
        """Encodes input features to logits.

        Args:
            features (dict): Dictionary with input features.

        Returns:
            torch.Tensor: The logits output from the model.
        """
        if features is None:
            return None
        outputs = self.model(input_ids=features['input_ids'],
                             attention_mask=features['attention_mask'],
                             position_ids=features['position_ids'] if 'position_ids' in features.keys() else None,
                             output_hidden_states=True)
        # if 'labels' in features.keys():
        #     _, max_indices = torch.max(features['labels'], dim=1)
        #     predict_indices = max_indices - 1
        #     logits = [outputs.logits[i, predict_indices[i], :] for i in range(outputs.logits.shape[0])]
        #     logits = torch.stack(logits, dim=0)
        #     scores = logits[:, self.yes_loc]
        # else:
        scores = outputs.logits[:, -1, self.yes_loc]
        return scores.contiguous()
