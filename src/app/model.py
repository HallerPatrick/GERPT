import sys
from typing import Optional

sys.path.append("..")


import torch
import torch.nn.functional as F

from src.models import RNNModel


def load_model(model_path):
    model = RNNModel.load_from_checkpoint(model_path)
    model.eval()
    return model


def generate(model, temp: float, seed: Optional[str], chars: int):
    """Generation tasks for a given seed text"""

    generated_output = seed if seed else ""

    inp = (
        model.dictionary.tokenize_line(
            list(generated_output),
            id_type=torch.int64,
            return_tensor="pt",
        )["source"]
        .unsqueeze(dim=2)
        .to(model.device)
    )

    generated_tokens = []

    with torch.no_grad():
        model.eval()

        for i in range(chars):
            hidden = model.init_hidden(1)
            output, hidden = model(inp, hidden)

            output = output[-1]

            if temp == 0.0:
                output = F.softmax(output, dim=0).cpu()
                # Just get highest confidence
                ngram_idx = torch.argmax(output)
                # Get ngram word
                token = model.dictionary.get_item_for_index(ngram_idx.item())
            else:
                output = F.log_softmax(output, dim=0)

                # Remove all UNK tokens for ngram > 2
                # if model.ngrams > 2:
                #     output = torch.index_select(output, 0, token_idxs).cpu()

                word_weights = output.squeeze().div(temp).exp().cpu()

                # multinomial over all tokens
                ngram_idx = torch.multinomial(word_weights, 1)[0]

                ngram_order = model.dictionary.get_ngram_order(ngram_idx.item())

                token = model.dictionary.get_item_for_index(ngram_idx.item())

            generated_tokens.append(token)

            # Append to generated sequence
            generated_output = generated_output + token

            # Use last 200 chars as sequence for new input
            inp = (
                model.dictionary.tokenize_line(
                    list(generated_output[-200:]),
                    id_type=torch.int64,
                    return_tensor="pt",
                )["source"]
                .unsqueeze(dim=2)
                .to(model.device)
            )
    return generated_tokens
