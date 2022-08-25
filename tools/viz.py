import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import StrMethodFormatter, NullFormatter


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
