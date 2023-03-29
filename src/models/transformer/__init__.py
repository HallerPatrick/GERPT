from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.models.transformer.configuration_transformer import TransformerConfig
from src.models.transformer.modelling_transformer import TransformerTransformer
from src.models.transformer.tokenization_transformer import NGMETokenizer

AutoConfig.register("ngme-transformer", TransformerConfig)
AutoTokenizer.register(TransformerConfig, NGMETokenizer)
AutoModel.register(TransformerConfig, TransformerTransformer)

