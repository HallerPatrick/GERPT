import argparse
import ast
import json

import wandb


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str)
    parser.add_argument("--values", type=str)

    args = parser.parse_args()

    run = wandb.Api().run(args.id)
    
    for k, v in ast.literal_eval(args.values).items():
        run.summary[k] = v

    run.update()
