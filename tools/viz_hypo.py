import json
import statistics

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

all_runs = pd.read_csv("project_all.csv")

fig, ax = plt.subplots(2, 3, sharex='col')

# common axis labels
fig.supxlabel("N-Gram Count")
fig.supylabel("f1-score (micro)")

for i, label in enumerate([("fixed-lstm", "b"), ("fixed-model", "g")]):
    
    runs = all_runs.loc[all_runs.hypo_type.isin([label[0], "baseline"])]
    
    df = pd.concat([ runs.ngram, runs.summary ], axis=1)
    df = df.sort_values("ngram")
    
    ngram_scores = {ngram: {} for ngram in range(1, max(df["ngram"]) + 1)}

    for _, row in df.iterrows():

        summary = row.summary
        
        summary = json.loads(summary.replace("\'", "\""))

        if "ner" not in ngram_scores[row.ngram]:
            ngram_scores[row.ngram]["ner"] = {"scores":[summary["ner/f1-score"]]}
        else:
            ngram_scores[row.ngram]["ner"]["scores"].append(summary["ner/f1-score"])

        if "upos" not in ngram_scores[row.ngram]:
            ngram_scores[row.ngram]["upos"] = {"scores":[ summary["upos/f1-score"] ]}
        else:
            ngram_scores[row.ngram]["upos"]["scores"].append(summary["upos/f1-score"])

        if "class" not in ngram_scores[row.ngram]:
            ngram_scores[row.ngram]["class"] = {"scores":[ summary["class/f1-score"] ]}
        else:
            ngram_scores[row.ngram]["class"]["scores"].append(summary["class/f1-score"])

    
    for _ngram in ngram_scores.keys():
        for task in ngram_scores[_ngram]:
            ngram_scores[_ngram][task]["stddev"] = statistics.stdev(ngram_scores[_ngram][task]["scores"])
            ngram_scores[_ngram][task]["mean"] = statistics.mean(ngram_scores[_ngram][task]["scores"])
        
    mean_ner = [scores["ner"]["mean"] for scores in ngram_scores.values()]
    stddev_ner = [scores["ner"]["stddev"] for scores in ngram_scores.values()]
    ngrams = [key for key in ngram_scores.keys()]
    ax[i, 0].plot(ngrams, mean_ner, color=label[1])
    ax[i, 0].errorbar(ngrams, mean_ner, stddev_ner, linestyle='-', fmt='o', color=label[1])

    ax[i, 0].set_title(f"NER: {label[0]}")
    # # Show only full integer ngrams
    ax[i, 0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


    mean_upos = [scores["upos"]["mean"] for scores in ngram_scores.values()]
    stddev_upos = [scores["upos"]["stddev"] for scores in ngram_scores.values()]
    ax[i, 1].plot(ngrams, mean_upos, color=label[1])
    ax[i, 1].errorbar(ngrams, mean_upos, stddev_upos, linestyle='-', fmt='o', color=label[1])
    ax[i, 1].set_title(f"UPOS: {label[0]}")
    ax[i, 1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    
    mean_class = [scores["class"]["mean"] for scores in ngram_scores.values()]
    stddev_class = [scores["class"]["stddev"] for scores in ngram_scores.values()]
    ax[i, 2].plot(ngrams, mean_class, color=label[1])
    ax[i, 2].errorbar(ngrams, mean_class, stddev_class, linestyle='-', fmt='o', color=label[1])
    ax[i, 2].set_title(f"CLASS: {label[0]}")
    ax[i, 2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    

plt.show()
