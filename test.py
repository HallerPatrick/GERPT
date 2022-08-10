
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
sentence1 = Sentence("""Bizarre horror movie filled with famous faces but stolen by Cristina Raines ( later of TV s " Flamingo Road ") as a pretty but somewhat unstable model with a gummy smile who is slated to pay for her attempted suicides by guarding the Gateway to Hell ! The scenes with Raines modeling are very well captured , the mood music is perfect , Deborah Raffin is charming as Cristina 's pal , but when Raines moves into a creepy Brooklyn Heights brownstone ( inhabited by a blind priest on the top floor ) , things really start cooking . The neighbors , including a fantastically wicked Burgess Meredith and kinky couple Sylvia Miles & Beverly D'Angelo , are a diabolical lot , and Eli Wallach is great fun as a wily police detective . The movie is nearly a cross-pollination of " Rosemary 's Baby " and " The Exorcist "-- but what a combination ! Based on the best-seller by Jeffrey Konvitz , " The Sentinel " is entertainingly spooky , full of shocks brought off well by director Michael Winner , who mounts a thoughtfully downbeat ending with skill . *** 1 / 2 from ****""")
sentence2 = Sentence("""For a movie that gets no respect there sure are a lot of memorable quotes listed for this gem . Imagine a movie where Joe Piscopo is actually funny ! Maureen Stapleton is a scene stealer . The Moroni character is an absolute scream . Watch for Alan " The Skipper " Hale jr. as a police Sgt""")


document_embeddings = NGMETransformerWordEmbeddings(
    "checkpoints/model.pt",
    layers='-1',
    layer_mean=False,
    vocab_file="checkpoints/model.pt/vocab.json",
)

document_embeddings.embed([sentence, sentence1, sentence2])

print(sentence.embedding)
print(sentence1.embedding)
print(sentence2.embedding)

# bert = NGMETransformerWordEmbeddings(
#     "bert-base-uncased"
# )


# bert.embed(sentence)
