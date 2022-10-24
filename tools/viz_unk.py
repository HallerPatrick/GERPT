import math
import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# pd.options.plotting.backend = "plotly"


def main():

    with open("unk_stats.json") as f:
        d = json.load(f)

    d2 = {
        "valid": [d[n]["valid"] for n in d.keys()],
        "unk": [d[n]["unk"] for n in d.keys()],
    }

    df = pd.DataFrame(d2, index=list(range(1, len(d.keys()) + 1)))
    ratios = list(
        map(lambda x: f"{round(x* 100, 2)} % unked", list(df["unk"] / df["valid"]))
    )

    ax = df.plot(kind="bar", stacked=True, use_index=True)

    labels = []

    for i, bars in enumerate(ax.containers):
        if i == 1:
            ax.bar_label(bars, ratios)

    plt.ylabel("Occurence valid and unk tokens")
    plt.xlabel("N-Gram")
    plt.title("N-Gram 'unking' in dataset")

    plt.show()

def main2():
    import plotly.graph_objects as go

    with open("unk_stats.json") as f:
        d = json.load(f)

    d2 = {
        "valid": [d[n]["valid"] for n in d.keys()],
        "unk": [d[n]["unk"] for n in d.keys()],
    }

    ratios = list(
        map(lambda x: f"{round(x* 100, 2)} % unked", [unk / valid for unk, valid in zip(d2["unk"], d2["valid"])])
    )

    ngrams = [1, 2, 3, 4]
    fig = go.Figure(
        data=[
            go.Bar(
                name=f"Valid", x=ngrams, y=d2["valid"]
            ),
            go.Bar(
                name=f"UNKed", x=ngrams, y=d2["unk"], text=ratios
            ),
        ]
    )

    fig.update_traces(textfont_size=24, textangle=0, textposition="outside", cliponaxis=False)

    fig.update_layout(
        barmode="stack",
        xaxis_title="N-Grams",
        yaxis_title="# Tokens",
        xaxis={
            "dtick": 1,
        },
        # font={"size": 24},
    )

    fig.show()

def viz_unking():

    df = pd.read_csv("./unk_thresholding.csv")

    import plotly.graph_objects as go

    ngrams = [1, 2, 3, 4]
    unks = [0, 30, 60, 90, 200]

    fig = go.Figure(
        data=[
            go.Bar(
                name=f"{unk}", x=ngrams, y=df["dict_size"][df["unk_threshold"] == unk]
            )
            for unk in unks
        ]
    )
    # Change the bar mode
    fig.update_layout(
        barmode="group",
        xaxis_title="N-Grams",
        yaxis_title="Dictionary Size",
        xaxis={
            "dtick": 1,
        },
        font={"size": 24},
        legend=dict(
            # yanchor="top", y=0.99, xanchor="left", x=0.01,
            title="Threshold"
        ),
    )
    fig.show()


if __name__ == "__main__":
    main2()
