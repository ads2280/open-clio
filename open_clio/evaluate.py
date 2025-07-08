import re
from typing import List

import anthropic
import pandas as pd
from langsmith import Client, wrappers
from openevals.llm import create_llm_as_judge
from prompts import (
    BEST_FIT,
    CATEGORY_RELEVANCE,
    DEDUPLICATE,
    EXCLUSIVE_FIT,
    HIERARCHICAL_FIT,
)

from open_clio.generate import load_config

config = load_config()

anthropic_client = wrappers.wrap_anthropic(anthropic.Anthropic())
client = Client()
dataset_name = "ds-granular-pseudoscience-68"
examples = list(client.list_examples(dataset_name=dataset_name))
print(f"Loaded {len(examples)} examples from dataset '{dataset_name}")

clusters_df = pd.read_csv("level_0_clusters.csv")
combined_df = pd.read_csv("combined.csv")
example_to_category = dict(zip(combined_df["full_example"], combined_df["category"]))
convo_to_cluster = dict(
    zip(combined_df["full_example"], combined_df["base_cluster_name"])
)
print(f"Loaded {len(example_to_category)} example->category mappings")
print(f"Loaded {len(convo_to_cluster)} convo->cluster mappings")


def category_relevance_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    """Evaluate if conversation fits its assigned category"""
    summary = reference_outputs["summary"]
    category = reference_outputs["category"]

    category_relevance_prompt = CATEGORY_RELEVANCE.format(
        summary=summary, category=category, partitions=config["partitions"]
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


def create_h_fit_evaluator(level: str):
    def hierarchical_fit_evaluator(
        inputs: dict, outputs: dict, reference_outputs: dict
    ):
        """Evaluate if conversation properly belongs under its assigned base cluster"""
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


def best_fit_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    all_base_clusters = clusters_df["name"].tolist()
    all_base_clusters_text = "\n".join(all_base_clusters)

    summary = reference_outputs["summary"]
    current_cluster = reference_outputs["clustering"]["level_0"]["name"]  # base only

    best_fit_prompt = BEST_FIT.format(
        summary=summary,
        current_cluster=current_cluster,
        all_base_clusters=all_base_clusters_text,
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


# evaluating redundancy by asking claude to check if a convo reasonably fits in 2+ clusters
def exclusive_fit_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    all_base_clusters = clusters_df["name"].tolist()
    all_base_clusters_text = "\n".join(
        [f"- {cluster}" for cluster in all_base_clusters]
    )

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


def unique_n_summary_evaluator(
    outputs: list[dict], reference_outputs: list[dict]
) -> dict:
    # get all unique clusters, same as best_fit_evaluator
    all_base_clusters = clusters_df["name"].tolist()
    all_base_clusters_text = "\n".join(
        [f"- {cluster}" for cluster in all_base_clusters]
    )
    total_base_clusters = len(all_base_clusters)

    prompt = DEDUPLICATE.format(
        all_base_clusters_text=all_base_clusters_text,
        total_base_clusters=total_base_clusters,
    )

    try:
        response = anthropic_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
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


def dummy_target(inputs):
    return {}


#  extract all available clustering levels from the data structure.
def get_clustering_levels(clustering_data: dict) -> List[str]:
    levels = []
    for key in clustering_data.keys():
        if key.startswith("level_"):
            levels.append(key)
    return sorted(levels)


# Setup all evaluators for multi-level evaluation
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


# run eval
limit = config["sample"] if config["sample"] is not None else 10
eval_results = client.evaluate(
    dummy_target,
    data=client.list_examples(dataset_name=config["dataset_name"], limit=limit),
    evaluators=prepare_evaluators(),
    summary_evaluators=[unique_n_summary_evaluator],
    experiment_prefix="exclusive-fit-eval",
    description="evaluating redundancy by asking claude to check if a convo reasonably fits in 2+ clusters",
    max_concurrency=2,
)


"""
{
  "summary": "debugging help with LangSmith SDK tracing for Python implementation",
  "category": "LangSmith product",
  "clustering": {
    "level_0": {
      "id": 5,
      "name": "Debug LangSmith Python SDK tracing integration errors"
    },
    "level_1": {
      "id": 2,
      "name": "Handle LangSmith SDK Integration and Tracing Issues"
    }
  }
}
"""
