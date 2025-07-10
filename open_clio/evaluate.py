import pandas as pd
from langsmith import Client, wrappers
from typing import List
import anthropic
from openevals.llm import create_llm_as_judge
import re
from open_clio.prompts import (
    BEST_FIT,
    PARTITION_RELEVANCE,
    DEDUPLICATE,
    EXCLUSIVE_FIT,
    HIERARCHICAL_FIT,
)
from open_clio.generate import load_config

anthropic_client = wrappers.wrap_anthropic(anthropic.Anthropic())
client = Client()
config = load_config()
dataset_name = config["dataset_name"]

combined_df = pd.read_csv("./.data/clustering_results/combined.csv")
clusters_df = pd.read_csv(
    "./.data/clustering_results/level_0_clusters.csv"
)  # base clusters
all_base_clusters = clusters_df["name"].tolist()
all_base_clusters_text = "\n".join([f"- {cluster}" for cluster in all_base_clusters])
total_base_clusters = len(all_base_clusters)
partitions = combined_df["partition"].unique().tolist()

examples = list(client.list_examples(dataset_name=dataset_name))


# Define evaluators
def best_fit_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """
    Evaluates whether a conversation is assigned to the most appropriate cluster, among all available base clusters.
    """
    summary = reference_outputs["summary"]
    current_cluster = reference_outputs["clustering"]["level_0"]["name"]  # base only

    best_fit_prompt = BEST_FIT.format(
        summary=summary,
        current_cluster=current_cluster,
        all_base_clusters_text=all_base_clusters_text,
    )

    evaluator = create_llm_as_judge(
        prompt=best_fit_prompt,
        model="anthropic:claude-sonnet-4-20250514",
        feedback_key="best-fit",
    )
    eval_result = evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )
    return eval_result


def exclusive_fit_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """
    Evaluates whether a conversation fits exclusively in its assigned cluster. A score of 0 means it could belong to multiple clusters.
    """
    summary = reference_outputs["summary"]
    current_cluster = reference_outputs["clustering"]["level_0"]["name"]  # base only

    redundancy_prompt = EXCLUSIVE_FIT.format(
        summary=summary,
        current_cluster=current_cluster,
        all_base_clusters_text=all_base_clusters_text,
    )

    evaluator = create_llm_as_judge(
        prompt=redundancy_prompt,
        model="anthropic:claude-sonnet-4-20250514",
        feedback_key="exclusive-fit",
    )

    eval_result = evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )

    return eval_result


def create_h_fit_evaluator(level: str):
    """
    Adaptively creates hierarchical fit evaluators for every level in the clustering hierarchy.
    """

    def hierarchical_fit_evaluator(
        inputs: dict, outputs: dict, reference_outputs: dict
    ):
        """
        Evaluates whether a conversation properly belongs in its assigned cluster at the specified level.
        """
        summary = reference_outputs["summary"]
        clustering = reference_outputs["clustering"]

        if level not in clustering:
            return {
                "key": f"cluster-{level}-relevance",
                "score": None,
                "comment": f"Level {level} not available for this example",
            }

        cluster_name = clustering[level]["name"]

        hierarchical_fit_prompt = HIERARCHICAL_FIT.format(
            summary=summary, current_cluster=cluster_name
        )

        evaluator = create_llm_as_judge(
            prompt=hierarchical_fit_prompt,
            model="anthropic:claude-sonnet-4-20250514",
            feedback_key=f"cluster-{level}-relevance",
        )

        eval_result = evaluator(
            inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
        )
        return eval_result

    return hierarchical_fit_evaluator


def unique_n_summary_evaluator(
    outputs: list[dict], reference_outputs: list[dict]
) -> dict:
    """
    Evaluates the uniqueness of base clusters.
    """
    prompt = DEDUPLICATE.format(
        all_base_clusters_text=all_base_clusters_text,
        total_base_clusters=total_base_clusters,
    )

    try:
        messages = [{"role": "user", "content": prompt}]
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.0,
            messages=messages,
        )

        response_text = response.content[0].text.strip()

        # extract final score,  "FINAL SCORE: X.XX"
        score = 0.0

        final_score_match = re.search(
            r"FINAL SCORE:\s*(\d*\.?\d+)", response_text, re.IGNORECASE
        )
        if final_score_match:
            score = float(final_score_match.group(1))
        else:
            score = -1
    except Exception as e:
        print({e})
    return {"key": "uniqueness_score", "score": float(score), "comment": response_text}


def category_relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """
    Evaluates whether a conversation is assigned to the correct partition.
    """
    summary = reference_outputs["summary"]
    partition = reference_outputs["partition"]

    category_relevance_prompt = PARTITION_RELEVANCE.format(
        summary=summary, partition=partition, partitions=partitions
    )

    evaluator = create_llm_as_judge(
        prompt=category_relevance_prompt,
        model="anthropic:claude-sonnet-4-20250514",
        feedback_key="category-relevance",
    )

    eval_result = evaluator(
        inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
    )
    return eval_result


def get_clustering_levels(clustering_data: dict) -> List[str]:
    levels = []
    for key in clustering_data.keys():
        if key.startswith("level_"):
            levels.append(key)
    return sorted(levels)


def prepare_evaluators():
    evaluators = []

    # Determine number of levels
    all_levels = set()
    for example in examples:
        clustering = example.outputs.get("clustering", {})
        levels = get_clustering_levels(clustering)
        all_levels.update(levels)

    print(f"Found clustering levels: {sorted(all_levels)}")

    # Create evaluators
    evaluators.append(category_relevance_evaluator)
    for level in sorted(all_levels):
        hierarchical_evaluator = create_h_fit_evaluator(level)
        evaluators.append(hierarchical_evaluator)
    evaluators.extend([exclusive_fit_evaluator, best_fit_evaluator])

    return evaluators


def dummy_target(inputs):
    return {}


def main():
    # check before updating the dataset
    while True:
        update_choice = (
            input(
                "Update the dataset with clustering results first? (y/n) - if you have not run evals on this dataset before, enter y: "
            )
            .lower()
            .strip()
        )
        if update_choice == "y":
            print("Updating dataset...")
            # Import and run update.py
            import subprocess
            import sys

            try:
                subprocess.run([sys.executable, "open_clio/update.py"], check=True)
                print("Dataset updated successfully!")
            except subprocess.CalledProcessError as e:
                print(f"Error updating dataset: {e}")
                print(
                    "Please run the clustering process first (python open_clio/generate.py)"
                )
                return
            break
        elif update_choice == "n":
            print("Skipping dataset update, proceeding with evaluation...")
            break
        else:
            print("Please enter 'y' or 'n'")

    # run eval
    print("Starting evaluation...")
    results = client.evaluate(
        dummy_target,
        data=client.list_examples(dataset_name=dataset_name),
        evaluators=prepare_evaluators(),
        summary_evaluators=[unique_n_summary_evaluator],
        experiment_prefix="all",
        description="all examples in the dataset, all evals",
        max_concurrency=2,
    )
    print(results)


if __name__ == "__main__":
    main()
