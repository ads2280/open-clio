from langsmith import Client
import pandas as pd
import re
from pathlib import Path
import ast


# RESULTS_DIR = Path(__file__).parent / "experiment_results"/"sonnet4.0-500-25-5-oai-embed"
# new_unthread_results/sonnet4/
RESULTS_DIR = Path(__file__).parent / "new_unthread_results"/"sonnet4" #Modify

client = Client()

base_df = pd.read_csv(str(RESULTS_DIR / "ADDHERE")) #Modify
l1_df = pd.read_csv(str(RESULTS_DIR /"ADDHERE")) #Modify
l2_df = pd.read_csv(str(RESULTS_DIR /"ADDHERE")) #Modify -- added for l2


l0_to_l1_clusters = {
    int(l0_id): {"level": 1, "id": row["cluster_id"], "name": row["name"]}
    for row in l1_df.to_dict(orient="records")
    for l0_id in re.findall(r"np\.int32\((\d+)\)", row["member_clusters"])
}

l1_to_l2_clusters = { #added for l2
    int(l1_id): {"level": 2, "id": row["cluster_id"], "name": row["name"]}
    for row in l2_df.to_dict(orient="records")
    for l1_id in ast.literal_eval(row["member_clusters"]) #changed b/c they aren't np but look like "[0,9,12]"
}

# examples = {str(e.id): e for e in client.list_examples(dataset_name="unthread-data", limit=300)}
limited_examples = client.list_examples(dataset_name="unthread-data", limit=300) #can increase/decrease
examples = {str(e.id): e for e in limited_examples}

results = {
   examples[row['example_id']].metadata['run_id']: { #TODO examples from unthread don't have metadata attribute
        "summary": row["summary"],
        "clusters": [
            {
                "level": 0,
                "id": row["base_cluster_id"],
                "name": row["base_cluster_name"],
            },
            l0_to_l1_clusters[row["base_cluster_id"]],
            l1_to_l2_clusters[l0_to_l1_clusters[row["base_cluster_id"]]["id"]] #added for l2
        ]
    }
    for row in base_df.to_dict(orient="records")
}

def dummy_target(inputs, metadata):
    return results[metadata["run_id"]]

client.evaluate( dummy_target, data=limited_examples, experiment_prefix="model:sonnet4-clustering:125/25/5")
