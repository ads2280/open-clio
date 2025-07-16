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
    rename_higher_level_clusters,
    llm,
    ASSIGN_CLUSTER_INSTR,
    RENAME_CLUSTER_INSTR,
    CRITERIA,
)
from logging import getLogger
from langsmith import Client
from langsmith import schemas as ls_schemas
from open_clio.internal.schemas import ExampleSummary
import random
import uuid
import time
import numpy as np
import operator

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
    current_level: int
    # for parallel assign_single_cluster
    cluster_id: int | None
    cluster_info: dict | None
    deduplicated: list[str] | None
    # for parallel rename_cluster_group
    hl_name: str | None
    member_cluster_ids: list | None
    level: int | None
    # for gen neighborhoods
    neighborhood_labels: list[int] | None
    target_clusters: int | None
    # results from this level needed for next
    cluster_assignments: Annotated[dict, merge_dict]
    parent_clusters: Annotated[dict, merge_dict]


class Config(TypedDict):
    max_concurrency: int | None


def prepare_next_level(
    state: ClusterState,
) -> (
    dict
):  # make parent clusters current clusters so we don't just repeat w/ current state
    return {
        "clusters": state["parent_clusters"],
        "current_level": state["current_level"] + 1,
    }


def next_level(state: ClusterState) -> Literal["prepare_next_level", "__end__"]:
    current_level = state["current_level"]
    print(f"\n\nDeciding whether to call next_level with current_level={current_level}, hierarchy={state['hierarchy']}\n\n")
    if current_level + 1 < len(state["hierarchy"]):
        return "prepare_next_level"  # which --> embed
    else:
        return "__end__"


def base_cluster(state: ClusterState) -> dict:
    curr_level = 0

    total_examples = len(state["summaries"])
    ratio = total_examples / state.get("total_examples", total_examples)
    scaled_hierarchy = [int(round(k * ratio)) for k in state["hierarchy"]]

    curr_k = scaled_hierarchy[curr_level]  # k for this partition at this level
    cluster_list = perform_base_clustering(
        state["summaries"], curr_k, state["partition"]
    )
    # need to track cluster IDs so switching to dict
    clusters = {cluster["id"]: cluster for cluster in cluster_list}
    return {"clusters": clusters, "current_level": curr_level}


def embed(state: ClusterState) -> dict:
    # dict[int, dict] where each key is a cluster id (int or uuid), and each value is a dict
    # with at least "id" and "name" fields (and possibly others like "description", "examples", etc.)
    current_clusters = state["clusters"]
    cluster_embeddings, cluster_ids = embed_cluster_descriptions(current_clusters)

    # add embed to clsuter to have in state
    for i, cluster_id in enumerate(cluster_ids):
        if cluster_id in current_clusters:
            current_clusters[cluster_id]["embeddings"] = cluster_embeddings[i]

    return {"clusters": current_clusters}


# def sample_examples(state: ClusterState) -> dict:
# wut


def generate_neighborhoods_step(state: ClusterState) -> dict:
    curr_level = state["current_level"]

    # setup: scale hierarchy based on partition size, get embeddings from clusters
    total_examples = len(state["summaries"])
    ratio = total_examples / state.get("total_examples", total_examples)
    scaled_hierarchy = [int(round(k * ratio)) for k in state["hierarchy"]]

    target_clusters = (
        scaled_hierarchy[curr_level]
        if curr_level < len(scaled_hierarchy)
        else scaled_hierarchy[-1]
    )

    cluster_embeddings = []
    cluster_ids = []
    for cluster_id, cluster_info in state["clusters"].items():
        if "embeddings" in cluster_info:
            cluster_embeddings.append(cluster_info["embeddings"])
            cluster_ids.append(cluster_id)

    cluster_embeddings = np.array(cluster_embeddings)

    # actual logic from generate_neighborhoods
    if len(cluster_embeddings) < 2:
        # to handle error where we only have 1 cluster
        neighborhood_labels = np.array([0])  # All in same neighborhood
    else:
        neighborhood_labels, _ = generate_neighborhoods(
            cluster_embeddings, len(cluster_embeddings), target_clusters
        )
        neighborhood_labels = np.array(neighborhood_labels)

    return {
        "neighborhood_labels": neighborhood_labels,
        "target_clusters": target_clusters,
    }


def propose_clusters(state: ClusterState) -> dict:
    current_clusters = state["clusters"]
    cluster_ids = list(current_clusters.keys())

    neighborhood_labels = state["neighborhood_labels"]
    k_neighborhoods = len(set(neighborhood_labels))
    target_clusters = state["target_clusters"]

    proposed = propose_clusters_from_neighborhoods(
        current_clusters,
        cluster_ids,
        neighborhood_labels,
        k_neighborhoods,
        target_clusters,
    )
    return {"proposed_clusters": proposed}


def dedup_clusters(state: ClusterState) -> dict:
    proposed = state["proposed_clusters"]
    target_clusters = state["target_clusters"]
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
    # TODO Extract the logic from assign_clusters_to_higher_level for a single cluster
    # Return {"cluster_assignment": {cluster_id: assigned_name}}

    cluster_id = state["cluster_id"]
    cluster_info_item = state["cluster_info"]
    deduplicated = state["deduplicated"]

    shuffled = deduplicated.copy()
    random.shuffle(shuffled)
    higher_level_text = "\n".join([f"<cluster>{name}</cluster>" for name in shuffled])

    # TODO messy
    assign_user_prompt = ASSIGN_CLUSTER_INSTR.format(
        higher_level_text=higher_level_text, specific_cluster=cluster_info_item
    )

    assign_assistant_prompt = """I understand. I'll evaluate the specific cluster and assign it to
the most appropriate higher-level cluster."""

    assign_user_prompt_2 = f"""
Now, here is the specific cluster to categorize:
<specific_cluster>
Name: {cluster_info_item["name"]}
Description: {cluster_info_item["description"]}
</specific_cluster>

Based on this information, determine the most appropriate higher-level
cluster and provide your answer as instructed."""

    assign_assistant_prompt_2 = """ 
Thank you, I will reflect on the cluster and categorize it most
appropriately within the LangChain support structure.  
<scratchpad>"""

    try:
        response = llm.invoke(
            [
                {"role": "user", "content": assign_user_prompt},
                {"role": "assistant", "content": assign_assistant_prompt},
                {"role": "user", "content": assign_user_prompt_2},
                {"role": "assistant", "content": assign_assistant_prompt_2},
            ],
            temperature=1.0,
        )
        content = str(response.content)
        if "<answer>" in content and "</answer>" in content:
            ans_start = content.find("<answer>") + 8
            ans_end = content.find("</answer>")
            assigned_cluster = content[ans_start:ans_end].strip()

            if assigned_cluster in deduplicated:
                assignment = assigned_cluster
            else:
                best_match = None
                for hl_name in deduplicated:
                    if (
                        assigned_cluster.lower() in hl_name.lower()
                        or hl_name.lower() in assigned_cluster.lower()
                    ):
                        best_match = hl_name
                        break
                assignment = best_match or deduplicated[0]
        else:
            assignment = deduplicated[0]
    except Exception as e:
        logger.error(f"Error assigning cluster {cluster_id}: {e}")
        assignment = deduplicated[0]

    return {"cluster_assignment": {cluster_id: assignment}}


def map_rename_clusters(state: ClusterState) -> list[Send]:
    print(f"\n\nmap_rename_clusters called with state={state}\n\n")
    current_clusters = state["clusters"]
    assignments = state["cluster_assignments"]
    level = len(state["clusters"])
    partition = state["partition"]

    cluster_groups = {}
    for cluster_id, assigned_hl in assignments.items():
        if assigned_hl not in cluster_groups:
            cluster_groups[assigned_hl] = []
        cluster_groups[assigned_hl].append(cluster_id)

    return [
        Send(
            "rename_cluster_group",
            {
                "hl_name": hl_name,
                "member_cluster_ids": member_ids,
                "current_clusters": current_clusters,
                "level": level,
                "partition": partition,
            },
        )
        for hl_name, member_ids in cluster_groups.items()
    ]


def rename_cluster_group(state: ClusterState) -> dict:
    hl_name = state["hl_name"]
    member_cluster_ids = state["member_cluster_ids"]
    current_clusters = state["current_clusters"]
    level = state["level"]
    partition = state["partition"]

    hl_id = uuid.uuid4()
    cluster_list = []
    total_size = 0
    for cluster_id in member_cluster_ids:
        cluster_info_item = current_clusters[cluster_id]
        cluster_list.append(f"<cluster>({cluster_info_item['name']})</cluster>")
        total_size += cluster_info_item.get(
            "size", cluster_info_item.get("total_size", 1)
        )

    cluster_list_text = "\n".join(cluster_list)

    renamingHL_user_prompt = RENAME_CLUSTER_INSTR.format(
        cluster_list_text=cluster_list_text, criteria=CRITERIA
    )

    renamingHL_assistant_prompt = """Sure, I will provide a clear, precise, and accurate summary and
name for this cluster. I will be descriptive and assume neither good nor
bad faith. Here is the summary, which I will follow with the name: <summary>"""

    try:
        response = llm.invoke(
            [
                {"role": "user", "content": renamingHL_user_prompt},
                {"role": "assistant", "content": renamingHL_assistant_prompt},
            ],
            temperature=1.0,
        )
        content = str(response.content)

        summary_end = content.find("</summary>")
        summary = (
            content[:summary_end].strip()
            if summary_end != -1
            else "Summary generation failed"
        )

        name_start = content.find("<name>") + 6
        name_end = content.find("</name>")
        name = (
            content[name_start:name_end].strip()
            if name_start != -1 and name_end != -1
            else f"Level {level} Cluster {hl_id}"
        )

    except Exception as e:
        logger.error(f"Error renaming cluster {hl_name}: {e}")
        name = f"Level {level} Cluster {hl_id}"
        summary = "Summary generation failed"

    parent_cluster = {
        "name": name,
        "description": summary,
        "member_clusters": member_cluster_ids,
        "total_size": total_size,
        "size": len(member_cluster_ids),
        "partition": partition,
    }

    logger.info(
        f"Level {level} Cluster {hl_id}: {name} ({len(member_cluster_ids)} sub-clusters, {total_size} total items)"
    )

    return {
        "parent_clusters": {hl_id: parent_cluster},
        "current_level": state["level"] + 1,
    }


def map_rename_clusters(state: ClusterState) -> list[Send]:
    current_clusters = state["clusters"]
    assignments = state["cluster_assignments"]
    level = state["current_level"]
    partition = state["partition"]

    # Group clusters by their assigned higher-level cluster name
    cluster_groups = {}
    for cluster_id, assigned_hl in assignments.items():
        if assigned_hl not in cluster_groups:
            cluster_groups[assigned_hl] = []
        cluster_groups[assigned_hl].append(cluster_id)

    return [
        Send(
            "rename_cluster_group",
            {
                "hl_name": hl_name,
                "member_cluster_ids": member_ids,
                "current_clusters": current_clusters,
                "level": level,
                "partition": partition,
            },
        )
        for hl_name, member_ids in cluster_groups.items()
    ]


cluster_builder = StateGraph(ClusterState)
cluster_builder.add_node(base_cluster)
cluster_builder.add_node(embed)
cluster_builder.add_node(generate_neighborhoods_step)
cluster_builder.add_node(propose_clusters)
cluster_builder.add_node(dedup_clusters)
cluster_builder.add_node(assign_single_cluster)
cluster_builder.add_node(rename_cluster_group)
cluster_builder.add_node(prepare_next_level)

cluster_builder.set_entry_point("base_cluster")
cluster_builder.add_edge("base_cluster", "embed")
cluster_builder.add_edge("embed", "generate_neighborhoods_step")
cluster_builder.add_edge("generate_neighborhoods_step", "propose_clusters")
cluster_builder.add_edge("propose_clusters", "dedup_clusters")
cluster_builder.add_conditional_edges(
    "dedup_clusters", map_assign_clusters, ["assign_single_cluster"]
)
cluster_builder.add_conditional_edges(
    "assign_single_cluster", map_rename_clusters, ["rename_cluster_group"]
)
cluster_builder.add_conditional_edges("rename_cluster_group", next_level)
cluster_builder.add_edge("prepare_next_level", "embed") 
cluster_graph = cluster_builder.compile()


class State(TypedDict):
    dataset_name: str
    sample: int | float | None
    hierarchy: Annotated[list[int], operator.add]
    partitions: dict | None
    summary_prompt: str | None
    examples: Annotated[list[ls_schemas.Example], operator.add]
    summaries: Annotated[list[ExampleSummary], lambda l, r: l + r]
    total_examples: Annotated[int, lambda l, r: max(l, r)]
    clusters: Annotated[dict, merge_dict]


def load_examples(state: State) -> dict:
    partitions = state["partitions"]
    hierarchy = state["hierarchy"]
    dataset_name = state["dataset_name"]

    print(f"load_examples: hierarchy = {hierarchy}")

    if not hierarchy:
        logger.error("hierarchy is empty!")
        print("hierarchy is empty!")
        raise ValueError("hierarchy cannot be empty")

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

# TODO figure out how to tqdm with langgraph

async def summarize(state: State) -> dict:
    example = state["example"]
    print(f"Processing example {example.id}")  # debug
    summary = await summarize_example(
        example,
        state["partitions"],
        state.get("summary_prompt", DEFAULT_SUMMARY_PROMPT),
    )
    print(f"summary: {summary}")  # debug
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

    print(f"Partitions: {list(set(summaries_by_partition.keys()))}")

    sends = []
    for partition, cat_summaries in summaries_by_partition.items():
        example_ids = [s["example_id"] for s in cat_summaries]
        partition_examples = [e for e in state["examples"] if e.id in example_ids]
        sends.append(
            Send(
                "cluster_partition",
                {
                    "partition": partition,
                    "examples": partition_examples,
                    "summaries": cat_summaries,
                    "hierarchy": state[
                        "hierarchy"
                    ],  # changed to keep original not partition specific
                    "total_examples": state["total_examples"],
                },
            )
        )
    return sends


partitioned_cluster_builder = StateGraph(State)
partitioned_cluster_builder.add_node(load_examples)
partitioned_cluster_builder.add_node(summarize)
partitioned_cluster_builder.add_node("cluster_partition", cluster_graph)
partitioned_cluster_builder.add_node("aggregate_summaries", {})
# partitioned_cluster_builder.add_node("aggregate_summaries", lambda x: x)

partitioned_cluster_builder.set_entry_point("load_examples")
partitioned_cluster_builder.add_conditional_edges(
    "load_examples", map_summaries, ["summarize"]
)
partitioned_cluster_builder.add_edge("summarize", "aggregate_summaries")
partitioned_cluster_builder.add_conditional_edges(
    "aggregate_summaries", map_partitions, ["cluster_partition"]
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

CONFIG_PATH = (
    Path(__file__).parents[1] / "configs/2_level.json"
)  # edited for anika's local

if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    results = asyncio.run(run_graph(**config))
    print(f"\n\nlen(results['clusters']): {len(results['clusters'])}")
    print(f"\n\nresults['clusters']: {results['clusters']}")

    #

    # add save clusters
