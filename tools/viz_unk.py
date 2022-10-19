import math
import json

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    
    with open("unk_stats.json") as f:
        d = json.load(f)

    print(d)

    d2 = {
            "valid": [d[n]["valid"] for n in d.keys()],
            "unk": [d[n]["unk"] for n in d.keys()]
    }


    df = pd.DataFrame(d2, index=list(range(1, len(d.keys())+1)))
    
    df.plot(kind="bar", stacked=True, use_index=True)

    plt.ylabel("Occurence valid and unk tokens")
    plt.xlabel("N-Gram")
    plt.title("N-Gram 'unking' in dataset")

    plt.show()


if __name__ == "__main__":
    main()
