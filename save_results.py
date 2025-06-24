import pandas as pd
import os
from datetime import datetime


def save_results(
    hierarchy, example_ids, cluster_labels, summaries, output_dir="clustering_results"
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) examples.csv for original data with base cluster assignments
    # basicallt what is already in chat-langchain-v3-selected dataset
    examples_data = []
    for i, (example_id, base_cluster_id) in enumerate(zip(example_ids, cluster_labels)):
        base_cluster_info = hierarchy["level_0"][base_cluster_id]
        examples_data.append(
            {
                "example_id": example_id,
                "summary": summaries[i],
                "base_cluster_id": int(base_cluster_id),
                "base_cluster_name": base_cluster_info["name"],
                "base_cluster_description": base_cluster_info["description"],
            }
        )

    df_examples = pd.DataFrame(examples_data)
    examples_file = f"{output_dir}/{timestamp}_examples.csv"
    df_examples.to_csv(examples_file, index=False)
    print(f"Saved examples: {examples_file} ({len(df_examples)} rows)")

    # 2) csv for each hierarchy level
    max_level = hierarchy["max_level"]

    for level in range(max_level + 1):
        level_data = []
        level_clusters = hierarchy[f"level_{level}"]

        for cluster_id, info in level_clusters.items():
            level_data.append(
                {
                    "cluster_id": cluster_id,
                    "name": info["name"],
                    "description": info["description"],
                    "size": info["size"],
                    "total_size": info.get("total_size", info["size"]),
                    "member_clusters": str(info.get("member_clusters", []))
                    if level > 0
                    else "",
                    "sample_summaries": str(info.get("summaries", [])[:3])
                    if level == 0
                    else "",
                }
            )

        df_level = pd.DataFrame(level_data)
        level_file = f"{output_dir}/{timestamp}_level_{level}_clusters.csv"
        df_level.to_csv(level_file, index=False)
        print(f"Saved level {level}: {level_file} ({len(df_level)} rows)")

    return {
        "examples": df_examples,
        **{
            f"level_{i}": pd.read_csv(
                f"{output_dir}/{timestamp}_level_{i}_clusters.csv"
            )
            for i in range(max_level + 1)
        },
    }
