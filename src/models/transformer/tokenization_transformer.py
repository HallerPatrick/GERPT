import json
from typing import Optional, Tuple

from transformers import PreTrainedTokenizer
from transformers import PretrainedConfig



class NGMETokenizer(PreTrainedTokenizer):

    vocab_file_name = "vocab.json"

    def __init__(self, vocab_file, config_file: PretrainedConfig, **kwargs):

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        self.decoder = {v: k for k, v in self.encoder.items()}
        super().__init__(**kwargs)

    @property
    def vocab_size(self):
        return len(self.encoder)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.encoder.get(token, self.encoder.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.decoder.get(index)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return super().save_vocabulary(save_directory, filename_prefix)
