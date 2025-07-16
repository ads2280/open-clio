import warnings
from collections import defaultdict
from typing import Literal
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from open_clio.generate import (
    DEFAULT_SUMMARIZATION_CONCURRENCY,
    validate_hierarchy,
    summarize_example,
    perform_base_clustering,
    embed_cluster_descriptions,
    generate_neighborhoods,
    propose_clusters_from_neighborhoods,
    deduplicate_proposed_clusters,
    assign_clusters_to_higher_level,
    rename_higher_level_clusters
)
from logging import getLogger
from langsmith import Client
from langsmith import schemas as ls_schemas
from open_clio.internal.schemas import ExampleSummary

DEFAULT_SUMMARY_PROMPT = """summarize por favor: {{example}}"""

logger = getLogger(__name__)


def merge_dict(l: dict, r: dict) -> dict:
    return {**l, **r}


class ClusterState(TypedDict):
    partition: str
    hierarchy: list[int]
    clusters: Annotated[dict[int, dict], merge_dict]
    examples: list[ls_schemas.Example]
    summaries: Annotated[list[ExampleSummary], lambda l, r: l + r]
    total_examples: int
    proposed_clusters: list[str] | None
    deduplicated_clusters: list[str] | None


class Config(TypedDict):
    max_concurrency: int | None


def next_level(state: ClusterState) -> Literal["embed", "__end__"]:
    if len(state["clusters"]) < len(state["hierarchy"]):
        return "embed"
    else:
        return "__end__"


def base_cluster(state: ClusterState) -> dict:
    curr_level = len(state["clusters"])
    curr_k = state["hierarchy"][curr_level]  # k for this partition at this level
    clusters = perform_base_clustering(state["summaries"], curr_k, state["partition"])
    return {"clusters": clusters}


def embed(state: ClusterState) -> dict:
    # clusters is a dict[int, dict] where each key is a cluster id (int or uuid), and each value is a dict
    # with at least "id" and "name" fields (and possibly others like "description", "examples", etc.)
    for id, info in state["clusters"].items():
        embeddings = embed_cluster_descriptions(info.get("description", ""))
        state["clusters"][id]["embeddings"] = embeddings
    return {"clusters": state["clusters"]}


def sample_examples(state: ClusterState) -> dict:
    ...
    # wut


def map_neighborhoods(state: ClusterState) -> list[Send]:
    return [
        Send(
            "generate_neighborhood",
            {
                "embeddings": state["clusters"][id]["embeddings"],
                "num_clusters": state["hierarchy"][len(state["clusters"])],
                "target_clusters": state["hierarchy"][len(state["clusters"]) + 1],
            },
        )
        for id in state["clusters"]
    ]


def generate_neighborhood(state: ClusterState) -> dict:
    cluster_embeddings = state["embeddings"]
    num_clusters = state["num_clusters"]
    target_clusters = state["target_clusters"]
    neighborhood_labels, _ = generate_neighborhoods(
        cluster_embeddings, num_clusters, target_clusters
    )
    return {"neighborhood_labels": neighborhood_labels}


def propose_clusters(state: ClusterState) -> dict:
    current_clusters = state["clusters"]
    cluster_ids = list(current_clusters.keys())
    neighborhood_labels = state["neighborhood_labels"]
    k_neighborhoods = len(set(neighborhood_labels))
    target_clusters = state["hierarchy"][len(current_clusters)]

    proposed = propose_clusters_from_neighborhoods(
        current_clusters,
        cluster_ids,
        neighborhood_labels,
        k_neighborhoods,
        target_clusters,
    )
    return {"proposed_clusters": proposed}   # just added to state?

def dedup_clusters(state: ClusterState) -> dict:
    proposed = state["proposed_clusters"]
    target_clusters = state["hierarchy"][len(state["clusters"])]
    deduplicated = deduplicate_proposed_clusters(proposed, target_clusters)
    return {"deduplicated_clusters": deduplicated}


def map_assign_clusters(state: ClusterState) -> list[Send]:
    current_clusters = state["clusters"]
    deduplicated = state["deduplicated_clusters"]
    return [
        Send(
            "assign_single_cluster",
            {
                "cluster_id": cluster_id,
                "cluster_info": cluster_info,
                "deduplicated": deduplicated,
            },
        )
        for cluster_id, cluster_info in current_clusters.items()
    ]

def assign_single_cluster(state: ClusterState) -> dict:
    # TODO - make logic from assign_clusters_to_higher_level for a single cluster
    # return {"cluster_assignment": {cluster_id: assigned_name}}
    # can decide whether to implement here or in generate.py
    pass

def map_rename_clusters(state: ClusterState) -> list[Send]:
    pass
# same as above

def rename_cluster_group(state: ClusterState) -> dict:
    pass

def rename_parent_clusters(state: ClusterState) -> dict: 
    current_clusters = state["clusters"]
    assignments = state["cluster_assignments"]
    level = len(state["clusters"])
    partition = state["partition"]
    parent_clusters = rename_higher_level_clusters(current_clusters, assignments, level, partition)
    return {"parent_clusters": parent_clusters}


def map_rename_clusters(state: ClusterState) -> list[Send]:
    pass
cluster_builder = StateGraph(ClusterState)
cluster_builder.add_node(base_cluster)
cluster_builder.add_node(embed)
cluster_builder.add_node(generate_neighborhood)
cluster_builder.add_node(propose_clusters)
cluster_builder.add_node(dedup_clusters)
cluster_builder.add_node(assign_single_cluster)
cluster_builder.add_node(rename_parent_clusters)

cluster_builder.add_edge("base_cluster", "embed")
cluster_builder.add_condition_edges(
    "embed", map_neighborhoods, ["generate_neighborhood"]
)
cluster_builder.add_edge("generate_neighborhood", "propose_clusters")
cluster_builder.add_edge("propose_clusters", "dedup_clusters")
# TODO: Parallelize -> in generate
cluster_builder.add_conditional_edges(
    "dedup_clusters", map_assign_clusters, ["assign_single_cluster"]
)
# TODO: Parallelize -> in generate
cluster_builder.add_conditional_edges(
    "assign_single_cluster", map_rename_clusters, ["rename_cluster_group"]
)
cluster_builder.add_conditional_edges("rename_parent_clusters", next_level)
cluster_graph = cluster_builder.compile()


class State(TypedDict):
    dataset_name: str
    sample: int | float | None
    hierarchy: list[int]
    partitions: dict | None
    summary_prompt: str | None
    examples: list[ls_schemas.Example]
    summaries: Annotated[list[ExampleSummary], lambda l, r: l + r]
    total_examples: int
    clusters: Annotated[dict, merge_dict]


def load_examples(state: State) -> dict:
    partitions = state["partitions"]
    hierarchy = state["hierarchy"]
    dataset_name = state["dataset_name"]
    if partitions is not None:
        num_partitions = len(partitions.keys())
        num_top_level_clusters = hierarchy[-1]
        if num_partitions != num_top_level_clusters:
            warnings.warn(
                f"Number of partitions ({num_partitions}) does not match number of "
                f"top-level clusters ({num_top_level_clusters})"
            )

    # load data
    logger.info(f"Loading and summarizing examples from '{dataset_name}' dataset")
    print(f"Loading dataset '{dataset_name}'...")

    client = Client()
    examples = list(
        client.list_examples(
            dataset_name=dataset_name,
            limit=state["sample"] if state.get("sample") else None,
        )
    )
    total_examples = len(examples)
    validate_hierarchy(hierarchy, total_examples)  # Gives you an option to quit

    logger.info(f"Loaded {total_examples} total examples, generating summaries...")
    print(f"Loaded {total_examples} examples, generating summaries...")
    return {"total_examples": total_examples, "examples": examples}


async def summarize(state: State) -> dict:
    example = state["example"]
    summary = await summarize_example(
        example,
        state["partitions"],
        state.get("summary_prompt", DEFAULT_SUMMARY_PROMPT),
    )
    return {"summaries": [summary]}


def map_summaries(state: State) -> list[Send]:
    return [
        Send(
            "summarize",
            {
                "example": e,
                "partitions": state["partitions"],
                "summary_prompt": state.get("summary_prompt"),
            },
        )
        for e in state["examples"]
    ]


def map_partitions(state: State) -> list[Send]:
    summaries_by_partition = defaultdict(list)
    for summary in state["summaries"]:
        if summary:
            summaries_by_partition[summary["partition"]].append(summary)

    logger.info(
        f"The dataset contains the following partitions: {list(summaries_by_partition)}"
    )
    print(f"Partitions: {list(summaries_by_partition.keys())}")

    sends = []
    for partition, cat_summaries in summaries_by_partition.items():
        example_ids = [s["example_id"] for s in cat_summaries]
        partition_examples = [e for e in state["examples"] if e.id in example_ids]
        # TODO update based on partition size - done
        ratio = len(cat_summaries) / state["total_examples"]
        hierarchy = [
            int(round(k * ratio)) for k in state["hierarchy"]
        ]  # make partition-specific hierarchy
        sends.append(
            Send(
                "cluster_partition",
                {
                    "partition": partition,
                    "examples": partition_examples,
                    "summaries": cat_summaries,
                    "hierarchy": hierarchy,
                },
            )
        )
    return sends


partitioned_cluster_builder = StateGraph(State)
partitioned_cluster_builder.add_node(load_examples)
partitioned_cluster_builder.add_node(summarize)
partitioned_cluster_builder.add_node("cluster_partition", cluster_graph)

partitioned_cluster_builder.set_entry_point("load_examples")
partitioned_cluster_builder.add_conditional_edges(
    "load_examples", map_summaries, ["summarize"]
)
partitioned_cluster_builder.add_conditional_edges(
    "summarize", map_partitions, ["cluster_partition"]
)
partitioned_cluster_builder.add_edge("cluster_partition", END)
partitioned_cluster_graph = partitioned_cluster_builder.compile()


async def run_graph(
    dataset_name: str,
    hierarchy: list,
    summary_prompt: str,  # TODO
    *,
    save_path: str | None = None,
    partitions: dict | None = None,
    sample: int | None = None,
    max_concurrency: int = DEFAULT_SUMMARIZATION_CONCURRENCY,
):
    results = await partitioned_cluster_graph.ainvoke(
        {
            "dataset_name": dataset_name,
            "hierarchy": hierarchy,
            "partitions": partitions,
            "sample": sample,
        },
        config={"summary_prompt": summary_prompt, "max_concurrency": max_concurrency},
    )
    return results


import json
from pathlib import Path
import asyncio

CONFIG_PATH = Path(__file__).parents[1] / ".data" / "config.json"

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    config["summary_prompt"] = "Summarize the following examples {{example}}"
    config["partitions"] = {}
    results = asyncio.run(run_graph(**config, config={"max_concurrency": 10}))
    print(results["clusters"])
