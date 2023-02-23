from typing import Optional

import torch
import torch.nn.functional as F

from src.args import argparse_generate
from src.models import RNNModel

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
    idx = inp[0][0].detach()
    sample_text = model.dictionary.get_item_for_index(idx.item())

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
                word = model.dictionary.get_item_for_index(ngram_idx.item())
            else:
                output = F.log_softmax(output, dim=0)

                # Remove all UNK tokens for ngram > 2
                # if model.ngrams > 2:
                #     output = torch.index_select(output, 0, token_idxs).cpu()

                word_weights = output.squeeze().div(temp).exp().cpu()

                # multinomial over all tokens
                ngram_idx = torch.multinomial(word_weights, 1)[0]

                ngram_order = model.dictionary.get_ngram_order(ngram_idx.item())

                ngrams_idxs = [ngram_idx]

                if model.dictionary.ngme == "sparse":
                    for i in range(1, ngram_order):
                        ngram_subset = torch.index_select(
                            word_weights,
                            0,
                            torch.tensor(
                                list(model.dictionary.ngram2idx2word[i].keys())
                            ),
                        )

                        ngrams_idxs.append(torch.multinomial(ngram_subset, 1)[0])

                word = "".join(
                    list(
                        reversed(
                            [
                                model.dictionary.get_item_for_index(idx.item())
                                for idx in ngrams_idxs
                            ]
                        )
                    )
                )

            # Append to generated sequence
            generated_output = generated_output + word
            sample_text = sample_text + "·" + word

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
    print(sample_text)


def main(args):

    model = RNNModel.load_from_checkpoint(args.model_path)

    model.temperature = args.temperature
    model.eval()

    for _ in range(args.num_iters):
        generate(model, args.temperature, args.seed, args.num_chars)
        print()
        print("="*80)
        print()


if __name__ == "__main__":
    main(argparse_generate())
