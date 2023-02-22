import json
import sys

sys.path.append("..")

from datasets.load import load_from_disk
import torch




def main():
    dataset = load_from_disk("data/tokenized_data")
    dictionary = torch.load("data/dict_data")

    stats = {n: {"unk": 0, "valid": 0} for n in range(1, dictionary.ngram+1)}
    unk_ids = [dictionary.ngram2word2idx[n]["<unk>"] for n in range(1, dictionary.ngram+1)]

    for row in dataset["train"]:
        for n, seq in enumerate(row["source"]):
            for idx in seq:
                if idx in unk_ids:
                    stats[n+1]["unk"] += 1
                else:
                    stats[n+1]["valid"] += 1


    
    with open("unk_stats.json", "w") as f:
        json.dump(stats, f)

if __name__ == "__main__":
    main()
