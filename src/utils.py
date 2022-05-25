import torch.nn.functional as F
from pytorch_lightning.callbacks.base import Callback


def display_text(dictionary, t):
    for a in t:
        print(repr(dictionary.idx2word[a.item()]), end="")
    print()


def display_input_n_gram_sequences(input, dictionary):
    for i in range(input.size()[0]):
        print(f"{i+1}-gram")
        display_text(dictionary, input[i])


def display_prediction(prediction, dictionary):
    prediction = F.softmax(prediction.view(-1), dim=0)
    preds = []
    for i, pred in enumerate(prediction):
        preds.append((i, pred.item()))

    preds = sorted(preds, key=lambda x: x[1], reverse=True)

    for p in preds:
        i, pred = p
        print("{:9}: {:.15f},".format(repr(dictionary.idx2word[i]), pred))
