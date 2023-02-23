import json
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter, NullFormatter

from src.loss import CrossEntropyLossSoft
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
                    output,
                    1,
                    torch.tensor(list(model.dictionary.ngram2idx2word[n].keys())),
                )
                n_gram_targets = torch.index_select(
                    targets,
                    1,
                    torch.tensor(list(model.dictionary.ngram2idx2word[n].keys())),
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

                idx_to_pred = [
                    (idx, pred.item()) for idx, pred in enumerate(total_output)
                ]
                idx_to_pred.sort(key=lambda x: x[1], reverse=True)

                r = None
                for rank, pred in enumerate(idx_to_pred):
                    if target_tokens[n - 1] == pred[0]:
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
                model.dictionary.tokenize_line(list(current_text), id_type=torch.int64)[
                    "source"
                ]
                .unsqueeze(dim=2)
                .to(model.device)
            )

            results[t] = t_results

    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":

    with open("results.json") as f:
        results = json.load(f)

    ngrams = 2

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    fig.subplots_adjust(hspace=0.5)

    settings = [(ax1, "entropy", "b", "Entropy"), (ax2, "loss", "g", "Loss"), (ax3, "rank", "r", "Rank (logscale)")]

    ngram_settings = [("-", "o"), ("--", "x"), (":", ".")]

    for ax, metric, color, label in settings:
        
        # Log scale for rank and also use non-scientific notation
        if metric in ["rank"]:
            ax.set_yscale("log", base=2)
            # ax.ticklabel_format(useOffset=False) #, style='plain')
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
            ax.yaxis.set_minor_formatter(NullFormatter())

        ax.grid(True)

        for ngram in range(1, ngrams + 1):

            if ngram == 1:
                continue

            x_axis = []
            y_axis = []

            for key, value in results.items():
                x_axis.append(value["char"])
                val = value[str(ngram)][metric]

                # if not val:
                #     val = 1

                y_axis.append(val)

            xs = list(range(0, len(x_axis)))
            ax.set_xticks(xs, x_axis, weight="bold")
            ax.set_ylabel(label)
    
            (plot,) = ax.plot(
                xs,
                y_axis,
                color,
                linestyle=ngram_settings[ngram-1][0],
                marker=ngram_settings[ngram-1][1],
                label=f"{ngram}-gram",
            )
            # Add the legend manually to the Axes.
            # ax.add_artist(legend)
            ax.legend()


    # plot
    plt.show()
