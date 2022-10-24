import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import seaborn as sns
import matplotx


def old():
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)

    fig.supxlabel("N-Gram Order")
    fig.supylabel("Label Weight")

    funcs = {
        "f(x) = x": lambda x: x,
        "f(x) = log(x+1)": lambda x: math.log(x+1),
        "f(x) = $x^2$": lambda x: x**2
    }


    for i, ( func_label, func ) in enumerate(funcs.items()):
        x_vals = list(range(1, 5))
        y_vals = list(map(func, list(range(1, 5))))
        y_vals = list(map(lambda x: x / sum(y_vals), y_vals))
        ax[i].set_title(func_label)
        sns.barplot(ax=ax[i], x=x_vals, y=y_vals)

    plt.show()


def new():

    from plotly.subplots import make_subplots

    import plotly.graph_objects as go


    funcs = {
        "$f(x) = x$": lambda x: x,
        "$f(x) = log(x+1)$": lambda x: math.log(x+1),
        "$f(x) = x^2$": lambda x: x**2
    }

    fig = make_subplots(rows=1, cols=3, subplot_titles=list(funcs.keys()))


    for i, ( func_label, func ) in enumerate(funcs.items()):
        x_vals = list(range(1, 5))
        y_vals = list(map(func, list(range(1, 5))))
        y_vals = list(map(lambda x: x / sum(y_vals), y_vals))

        fig.add_trace(
            go.Bar(name=func_label, x=x_vals, y=y_vals),
            row=1, col=i+1
        )


    fig.update_layout(
        title_text="",
        xaxis_title="N-Grams",
        yaxis_title="Weight Value",
        font=dict(size=22),
        showlegend=False
    )
    fig.show()


if __name__ == "__main__":
    new()

