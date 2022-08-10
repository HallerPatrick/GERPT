
from flair.data import Sentence
from flair.embeddings.token import TransformerWordEmbeddings
from transformers import AutoConfig, AutoModel, AutoTokenizer
from src.models.flair_models import NGMETransformerWordEmbeddings
from src.models.transformer import TransformerTransformer, TransformerConfig, NGMETokenizer


# AutoConfig.register("ngme-transformer", TransformerConfig)
# AutoModel.register(TransformerConfig, TransformerTransformer)
# AutoTokenizer.register(TransformerConfig, NGMETokenizer)
#
# config = AutoConfig.from_pretrained("./checkpoints/model.pt/")
# model = AutoModel.from_pretrained("./checkpoints/model.pt/")
#
# tokenizer = AutoTokenizer.from_pretrained("./checkpoints/model.pt/", vocab_file="./checkpoints/model.pt/vocab.json")
#
# print(tokenizer._tokenize("Love"))
#
# print(tokenizer.encode("Love"))
#


sentence = Sentence("Love")

document_embeddings = NGMETransformerWordEmbeddings(
    "checkpoints/model.pt",
    vocab_file="checkpoints/model.pt/vocab.json",
)

bert = NGMETransformerWordEmbeddings(
    "gpt-2"
)

# document_embeddings.embed(sentence)

bert.embed(sentence)
