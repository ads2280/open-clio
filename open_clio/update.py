import pandas as pd
from langsmith import Client


def update_dataset(config):
    """
    Updates the dataset with clustering results, before running evals.
    """
    client = Client()
    dataset_name = config["dataset_name"]
    save_path = config.get("save_path", "./clustering_results")

    combined_df = pd.read_csv(f"{save_path}/combined.csv")
    
    # to compare example.id from langsmith with example_id from combined.csv, need to convert both to strings
    combined_df['example_id'] = combined_df['example_id'].astype(str)
    combined_df.set_index('example_id', inplace=True)

    # Respect sample limit from config if specified
    sample_limit = config.get("sample")
    if sample_limit is not None:
        print(f"Limiting dataset update to {sample_limit} samples as specified in config")
        examples = list(client.list_examples(dataset_name=dataset_name, limit=sample_limit))
    else:
        examples = list(client.list_examples(dataset_name=dataset_name))
    print(f"Found {len(examples)} examples to update")

    updates = []
    for example in examples:
        example_id_str = str(example.id)
        
        if example_id_str not in combined_df.index:
            print(f"Warning: Example {example_id_str} not found in combined.csv, skipping...")
            continue
            
        row = combined_df.loc[example_id_str]

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
