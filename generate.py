import json
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from src.args import argparse_generate
from src.loss import CrossEntropyLossSoft
from src.models import RNNModel
from src.models.ngme import soft_n_hot
from src.utils import DummyLogger


def analyze(model, seed_text: str, target: str):

    current_text = seed_text

    inp = (
        model.dictionary.tokenize_line(list(current_text), otf=True)["source"]
        .unsqueeze(dim=2)
        .to(model.device)
    )

    # nlll = torch.nn.NLLLoss()
    nlll = CrossEntropyLossSoft()

    results = {}

    with torch.no_grad():
        model.eval()

        # Measure metrics against the target sequence, so one iteration per char of target
        for t, c in enumerate(target):

            print(f"Current text: {current_text}")

            # Reset hidden
            hidden = model.init_hidden(1)
            output, hidden = model(inp, hidden)

            targets = (
                model.dictionary.tokenize_line(
                    list(current_text + c)[-len(current_text) :], otf=True
                )["source"]
                # .unsqueeze(dim=2)
                .to(model.device)
            )

            targets = soft_n_hot(targets, len(model.dictionary))
            
            t_results = {
                    "char": c
            }
            # Get all class of each n-gram
            for n in range(1, model.ngrams + 1):

                # Loss for each n-gram can be calculated by just using the subset of n-gram indexes
                # of the output and target
                n_gram_output = torch.index_select(
                    output, 1, torch.tensor(model.dictionary.ngram_indexes[n])
                )
                n_gram_targets = torch.index_select(
                    targets, 1, torch.tensor(model.dictionary.ngram_indexes[n])
                )

                n_gram_loss = nlll(n_gram_output, n_gram_targets)
                print(f"Loss {n}-gram: {n_gram_loss}")

                # Entropy, only based on output for new character
                out = F.softmax(n_gram_output[-1], dim=0).detach()
                entropy = Categorical(out).entropy()
                print(f"Entropy: {entropy}")

                ngram_idx = torch.argmax(out)

                target_idx = model.dictionary.word2idx[c]

                idx_to_pred = [(idx, pred.item())  for idx, pred in enumerate(out)]
                idx_to_pred.sort(key=lambda x: x[1], reverse=True)
                    
                r = None
                for rank, pred in enumerate(idx_to_pred):
                    if target_idx == pred[0]:
                        r = rank
                        print(f"Rank of {n}-gram '{c}': {rank}")

                t_results[n] = {
                        "loss": n_gram_loss.item(),
                        "entropy": entropy.item(),
                        "rank": r
                }

                # But not the rank, due to the loss of index if we subset output and target

            current_text = current_text + c

            inp = (
                model.dictionary.tokenize_line(list(current_text), otf=True)["source"]
                .unsqueeze(dim=2)
                .to(model.device)
            )

            results[t] = t_results

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


def generate(args):

    seed = "love "
    target = "is a burning ring"

    if args.type == "rnn":
        model = RNNModel.load_from_checkpoint(args.model)
    else:
        exit(-1)

    model.temperature = args.temperature
    model.eval()

    analyze(model, seed, target)

    # print(model.generate_text())


if __name__ == "__main__":
    generate(argparse_generate())
