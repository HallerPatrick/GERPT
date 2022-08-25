import pandas as pd 
import wandb

api = wandb.Api()

entity, project = "hallerpatrick", "gerpt"  # set to your entity and project 
runs = api.runs(entity + "/" + project) 

summary_list, config_list, name_list, hypo_types, ngram_list = [], [], [], [], []

for run in runs: 
    
    if "hypo" not in run.tags:
        continue

    run.tags.remove("hypo")

    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config = {k: v for k,v in run.config.items()
         if not k.startswith('_')}
    config_list.append(config)

    ngram_list.append(config["ngram"])

    # .name is the human-readable name of the run.
    name_list.append(run.name)
    if len(run.tags) == 1:
        hypo_types.append(run.tags[0])
    else:
        hypo_types.append("baseline")


runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "ngram": ngram_list,
    "name": name_list,
    "hypo_type": hypo_types
    })

runs_df.to_csv("project.csv")
