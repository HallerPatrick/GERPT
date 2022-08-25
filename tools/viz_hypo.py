import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

all_runs = pd.read_csv("project_all.csv")

fig, ax = plt.subplots(2, 3, sharex='col')

# common axis labels
fig.supxlabel("N-Gram Count")
fig.supylabel("f1-score (micro)")

for i, label in enumerate(["fixed-lstm", "fixed-model"]):
    
    runs = all_runs.loc[all_runs.hypo_type.isin([label, "baseline"])]
    
    ngram, ner, upos, clas = [], [], [], []
    
    df = pd.concat([ runs.ngram, runs.summary ], axis=1)
    df = df.sort_values("ngram")

    for _, row in df.iterrows():

        summary = row.summary
        
        summary = json.loads(summary.replace("\'", "\""))

        ner.append(summary["ner/f1-score"])
        upos.append(summary["upos/f1-score"])
        clas.append(summary["class/f1-score"])
        ngram.append(row.ngram)
    
    ax[i, 0].scatter(ngram, ner)
    ax[i, 0].plot(ngram, ner)
    ax[i, 0].set_title(f"NER: {label}")
    # Show only full integer ngrams
    ax[i, 0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax[i, 1].scatter(ngram, upos)
    ax[i, 1].plot(ngram, upos)
    ax[i, 1].set_title(f"UPOS: {label}")
    ax[i, 1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax[i, 2].scatter(ngram, clas)
    ax[i, 2].plot(ngram, clas)
    ax[i, 2].set_title(f"CLASS: {label}")
    ax[i, 2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))


plt.show()






