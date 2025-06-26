from langsmith import Client
import pandas as pd
import re
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "clustering_results"

client = Client()

base_df = pd.read_csv(str(RESULTS_DIR / "20250625_145851_examples.csv"))
l1_df = pd.read_csv(str(RESULTS_DIR /"20250625_145851_level_1_clusters.csv"))

l0_to_l1_clusters = {
    int(l0_id): {"level": 1, "id": row["cluster_id"], "name": row["name"]}
    for row in l1_df.to_dict(orient="records")
    for l0_id in re.findall(r"np\.int32\((\d+)\)", row["member_clusters"])
}

examples = {str(e.id): e for e in client.list_examples(dataset_name="open-clio-data-test")}

results = {
   examples[row['example_id']].metadata['run_id']: {
        "summary": row["summary"],
        "clusters": [
            {
                "level": 0,
                "id": row["base_cluster_id"],
                "name": row["base_cluster_name"],
            },
            l0_to_l1_clusters[row["base_cluster_id"]],
        ]
    }
    for row in base_df.to_dict(orient="records")
}


def dummy_target(inputs, metadata):
    return results[metadata["run_id"]]

client.evaluate( dummy_target, data="open-clio-data-test")

