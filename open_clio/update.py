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
    }

    # only add level_1 if the columns exist and have valid values
    if (
        "level_1_id" in row
        and "level_1_name" in row
        and pd.notna(row.get("level_1_id"))
        and pd.notna(row.get("level_1_name"))
    ):
        clustering["level_1"] = {
            "id": row.get("level_1_id"),
            "name": row.get("level_1_name"),
        }

    # only add level_2 if the columns exist and have valid values
    if (
        "level_2_id" in row
        and "level_2_name" in row
        and pd.notna(row.get("level_2_id"))
        and pd.notna(row.get("level_2_name"))
    ):
        clustering["level_2"] = {
            "id": row.get("level_2_id"),
            "name": row.get("level_2_name"),
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
