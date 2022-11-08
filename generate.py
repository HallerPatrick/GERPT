import json

import torch
import torch.nn.functional as F

from src.args import argparse_generate
from src.loss import CrossEntropyLossSoft
from src.models import RNNModel
from src.models.ngme import soft_n_hot


def calc_entropy(input_tensor):
    lsm = torch.nn.LogSoftmax()
    log_probs = lsm(input_tensor)
    probs = torch.exp(log_probs)
    p_log_p = log_probs * probs
    entropy = -p_log_p.mean()
    return entropy


def analyze(model, seed_text: str, target: str):
    """
    Running a forced teacher generation task and measuring the loss, rank of predicted tokens
    and entropy. The data is used to plot a timeline of all metrics over the predicted sequence.
    """

    current_text = seed_text

    inp = (
        model.dictionary.tokenize_line(list(current_text))["source"]
        .unsqueeze(dim=2)
        .to(model.device)
    )

    # nlll = torch.nn.NLLLoss()
    nlll = CrossEntropyLossSoft()
    # nlll = torch.nn.NLLLoss()

    results = {}

    hidden = None

    with torch.no_grad():
        model.eval()

         # Measure metrics against the target sequence, so one iteration per char of target
        for t, c in enumerate(target):

            print(f"Current text: {current_text}")

            # Reset hidden
            if not hidden:
                hidden = model.init_hidden(1)

            output, hidden = model(inp, hidden)

            targets = (
                model.dictionary.tokenize_line(
                    list(current_text + c)[-len(current_text) :]
                )["source"]
                # .unsqueeze(dim=2)
                .to(model.device)
            )

            target_tokens = targets[:, -1]

            targets = soft_n_hot(targets, len(model.dictionary), "exp")

            t_results = {"char": c}
            # Get all class of each n-gram
            for n in range(1, model.ngrams + 1):

                token = "".join(list(current_text + c)[-n:])
                
                # Loss for each n-gram can be calculated by just using the subset of n-gram indexes
                # of the output and target
                n_gram_output = torch.index_select(
                    output, 1, torch.tensor(list(model.dictionary.ngram2idx2word[n].keys()))
                )
                n_gram_targets = torch.index_select(
                    targets, 1, torch.tensor(list(model.dictionary.ngram2idx2word[n].keys()))
                )

                n_gram_targets[n_gram_targets != 0] = 1

                print(n_gram_output.size())
                print(n_gram_targets.size())

                n_gram_loss = nlll(n_gram_output, n_gram_targets)
                print(f"Loss {n}-gram: {n_gram_loss}")


                # TODO: Test this impl
                entropy = calc_entropy(n_gram_output[-1])
                
                # Entropy, only based on output for new character
                out = F.softmax(n_gram_output[-1], dim=0).detach()

                # entropy = Categorical(out).entropy()
                print(f"Entropy: {entropy}")

                target_idx = torch.argmax(out)
                total_output = F.log_softmax(output[-1], dim=0).detach()

                idx_to_pred = [(idx, pred.item()) for idx, pred in enumerate(total_output)]
                idx_to_pred.sort(key=lambda x: x[1], reverse=True)
                
                r = None
                for rank, pred in enumerate(idx_to_pred):
                    if target_tokens[n-1] == pred[0]:
                        r = rank + 1
                        print(f"Rank of {n}-gram '{c}': {rank}")
                        break
                else:
                    print(target_idx)
                    print(c)
                    print(idx_to_pred)

                t_results[n] = {
                    "loss": n_gram_loss.item(),
                    "entropy": entropy.item(),
                    "rank": r,
                }

                # But not the rank, due to the loss of index if we subset output and target

            current_text = current_text + c

            inp = (
                model.dictionary.tokenize_line(list(current_text), id_type=torch.int64)["source"]
                .unsqueeze(dim=2)
                .to(model.device)
            )

            results[t] = t_results

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)


def generate(model, temp: float, seed: str, chars: int):
    """Generation tasks for a given seed text"""

    generated_output = seed

    inp = (
        model.dictionary.tokenize_line(list(generated_output))["source"]
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
                            torch.tensor(list(model.dictionary.ngram2idx2word[i].keys()))
                        )

                        ngrams_idxs.append(torch.multinomial(ngram_subset, 1)[0])

                word = "".join(list(reversed([model.dictionary.get_item_for_index(idx.item()) for idx in ngrams_idxs])))

            # Append to generated sequence
            generated_output = generated_output + word 
            sample_text = sample_text + "·" + word

            # Use last 200 chars as sequence for new input
            inp = (
                model.dictionary.tokenize_line(list(generated_output))["source"]
                .unsqueeze(dim=2)
                .to(model.device)
            )
    print(sample_text)


def main(args):

#     seed = "Harry heard the sarcasm in his \
# voice, but he was not sure that anyone else did. \
# Opposite Harry, Tonks was entertaining Hermione and Ginny by \
# transforming her nose between mouthfuls. Screwing up her eyes each \
# time with the same pained expression she had worn back in Harry’s \
# bedroom, her nose swelled to a beaklike protuberance like Snape’s, \
# shrank to something resembling a button mushroom, and then \
# sprouted a great deal of hair from each nostril. Apparently this was a \
# regular mealtime entertainment,"
#     target = " because after a while Hermione and \
# Ginny started requesting their favorite noses."
    
    seed = "Love is"
    target = " a burning thing And "

    

    chars = 100

    if args.type == "rnn":
        model = RNNModel.load_from_checkpoint(args.model)
    else:
        print("Transformer not yet supported")
        exit(-1)

    model.temperature = args.temperature
    model.eval()

    if args.mode == "gen":
        for _ in range(args.num):
            generate(model, args.temperature, seed, chars)
            print("-" * 70)
    else:
        analyze(model, seed, target)


if __name__ == "__main__":
    main(argparse_generate())
