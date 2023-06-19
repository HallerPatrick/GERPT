from transformers import PretrainedConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CharFormerConfig(PretrainedConfig):
    model_type = "char_former"

    def __init__(
        self,
        vocab_size=384,
        hidden_size=512,
        num_hidden_layers=16,
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512,
        unk_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        dropout=0.55,
        intermediate_layer_predictions=True,
        tie_word_embeddings=False,
        use_ngme=False,
        **kwargs,
    ):
        super().__init__(
            unk_token_id=unk_token_id,
            pad_token_id_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.use_ngme = use_ngme
        self.vocab_size = vocab_size
        self.max_positional_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.n_layers = num_hidden_layers
        self.n_heads = num_attention_heads
        self.inner_linear = intermediate_size
        self.dropout = dropout
        self.unk_token_id = unk_token_id
        self.tied = tie_word_embeddings
        self.intermediate_layer_predictions = intermediate_layer_predictions
