from langsmith import Client, trace
import pandas as pd
import re
from pathlib import Path
import uuid
import ast

RESULTS_DIR = Path(__file__).parent / "new_unthread_results/sonnet4-3layers-125"
#20250627_144802_level_2_clusters.csv

client = Client()

base_df = pd.read_csv(str(RESULTS_DIR / "20250627_144802_examples.csv"))
l1_df = pd.read_csv(str(RESULTS_DIR /"20250627_144802_level_1_clusters.csv"))
l2_df = pd.read_csv(str(RESULTS_DIR /"20250627_144802_level_2_clusters.csv")) #Modify -- added for l2

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

results = {
    row['example_id']: {
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

project = client.create_project(f'experiment-{str(uuid.uuid4())[:8]}', reference_dataset_id="d253e939-b274-4699-99e1-abe65af56f8c")
print(project.name)
for example in client.list_examples(dataset_name="unthread-data", limit=5,):
    with trace(project_name=project.name,inputs=example.inputs, name='clio', client=client) as run:
        run.reference_example_id = example.id
        run.outputs = results[str(example.id)]

from langsmith import traceable
@traceable(client=client)
def foo(x):
    return x

foo(3)