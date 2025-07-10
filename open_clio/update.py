# todo edit this for multiple levels
import pandas as pd
from langsmith import Client
from open_clio.generate import load_config

client = Client()
config = load_config()
dataset_name = config["dataset_name"]

combined_df = pd.read_csv("./.data/clustering_results/combined.csv")

examples = list(client.list_examples(dataset_name=dataset_name))

updates = []
for idx, example in enumerate(examples):
    row = combined_df.iloc[idx]

    clustering = {
        "level_0": {
            "id": row.get("base_cluster_id"),
            "name": row.get("base_cluster_name"),
        },
        "level_1": {
            "id": row.get("top_cluster_id"),
            "name": row.get("top_cluster_name"),
        },
    }

    new_outputs = {
        "summary": row.get("summary"),
        "partition": row.get("partition"),
        "clustering": clustering,
    }

    updates.append(
        {
            "id": example.id,
            "inputs": example.inputs,
            "outputs": new_outputs,
            "metadata": example.metadata,
        }
    )

print(f"{len(updates)} updates created")

client.update_examples(updates=updates, dataset_name=dataset_name)
