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
import json
from pathlib import Path
import asyncio
import os
import pandas as pd
from collections import defaultdict

DEFAULT_SUMMARY_PROMPT = """summarize por favor: {{example}}"""

logger = getLogger(__name__)

def merge_dict(l: dict, r: dict) -> dict:
    result = l.copy()
    for key, value in r.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # merge the inner dicts for partitions otherwise only get results for 1 partition
            result[key] = {**result[key], **value}
        else:
            # overwrite like before
            result[key] = value
    return result

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
    member_cluster_infos: dict | None
    level: int | None
    # for gen neighborhoods
    neighborhood_labels: list[int] | None
    target_clusters: int | None
    # results from this level needed for next
    cluster_assignments: Annotated[dict, merge_dict]
    parent_clusters: Annotated[dict, merge_dict]
    # Track all clusters by level
    all_clusters_by_level: Annotated[dict, merge_dict]

class ClusterStateOutput(TypedDict):
    partition: str
    hierarchy: list[int]
    clusters: Annotated[dict[int, dict], merge_dict]
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
    member_cluster_infos: dict | None
    level: int | None
    # for gen neighborhoods
    neighborhood_labels: list[int] | None
    target_clusters: int | None
    # results from this level needed for next
    cluster_assignments: Annotated[dict, merge_dict]
    parent_clusters: Annotated[dict, merge_dict]
    # Track all clusters by level
    all_clusters_by_level: Annotated[dict, merge_dict]

class Config(TypedDict):
    max_concurrency: int | None

def prepare_next_level(state: ClusterState) -> dict:
    # save level n clusters before moving to level n+1
    current_level_clusters = {f"level_{state['current_level']}": state["clusters"]}
    existing_clusters = state.get("all_clusters_by_level", {})
    updated_clusters = {**existing_clusters, **current_level_clusters}

    if state["current_level"] == 0:
        next_clusters = state["clusters"]
    else:
        next_clusters = state["parent_clusters"]

    return {
        "clusters": next_clusters,
        "current_level": state["current_level"] + 1,
        "all_clusters_by_level": updated_clusters,
        "cluster_assignments": {},
        "parent_clusters": {},
        "proposed_clusters": None,
        "deduplicated_clusters": None,
        "neighborhood_labels": None,
        "target_clusters": None,
        "cluster_id": None,
        "cluster_info": None,
        "deduplicated": None,
        "hl_name": None,
        "member_cluster_ids": None,
        "member_cluster_infos": None,
        "level": None,
    }

def next_level(state: ClusterState) -> Literal["prepare_next_level", "__end__"]:
    current_level = state["current_level"]
    if current_level + 1 < len(state["hierarchy"]):
        return "prepare_next_level"  # which --> embed
    else:
        return "__end__"

def base_cluster(state: ClusterState) -> dict:
    curr_level = 0

    total_examples = len(state["summaries"])
    full_total_examples = state.get("total_examples", total_examples)

    if total_examples < full_total_examples:
        ratio = total_examples / full_total_examples
        partition_k = int(round(state["hierarchy"][0] * ratio))
    else:
        partition_k = state["hierarchy"][0]

    cluster_list = perform_base_clustering(
        state["summaries"], partition_k, state["partition"]
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

    # For hierarchical clustering, scale the hierarchy for partitions
    # The base level (level 0) is handled by base_cluster function
    # When we reach generate_neighborhoods_step, we're already at level 1+
    total_examples = len(state["summaries"])
    full_total_examples = state.get("total_examples", total_examples)

    if total_examples < full_total_examples:
        ratio = total_examples / full_total_examples
        target_clusters = int(round(state["hierarchy"][curr_level] * ratio))
    else:
        # this is the full dataset, use hierarchy[curr_level] directly
        target_clusters = state["hierarchy"][curr_level]

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

    return {"cluster_assignments": {cluster_id: assignment}}


def aggregate_assignments(state: ClusterState) -> dict:
    return state


def map_rename_clusters(state: ClusterState) -> list[Send]:
    current_clusters = state["clusters"]  # Get clusters from state
    assignments = state["cluster_assignments"]
    level = state["current_level"]  # prev "level": len(state["clusters"])
    partition = state["partition"]

    # Only process assignments for clusters that exist in current_clusters
    # fixes an error where we try to rename a cluster that doesn't exist at this level
    valid_assignments = {k: v for k, v in assignments.items() if k in current_clusters}

    # if len(valid_assignments) != len(assignments):
    #    print(
    #        f"WARNING: Filtered out {len(assignments) - len(valid_assignments)} assignments for clusters not in current level"
    #    )

    cluster_groups = {}
    for cluster_id, assigned_hl in valid_assignments.items():
        if assigned_hl not in cluster_groups:
            cluster_groups[assigned_hl] = []
        cluster_groups[assigned_hl].append(cluster_id)

    return [
        Send(
            "rename_cluster_group",
            {
                "hl_name": hl_name,
                "member_cluster_ids": member_ids,
                "level": level,
                "partition": partition,
                "member_cluster_infos": {
                    cluster_id: current_clusters[cluster_id]
                    for cluster_id in member_ids
                },  # specific cluster infos for one cluster
            },
        )
        for hl_name, member_ids in cluster_groups.items()
    ]


def rename_cluster_group(state: ClusterState) -> dict:
    hl_name = state["hl_name"]
    member_cluster_ids = state["member_cluster_ids"]
    member_cluster_infos = state["member_cluster_infos"]
    level = state["level"]
    partition = state["partition"]

    hl_id = uuid.uuid4()
    cluster_list = []
    total_size = 0
    for cluster_id in member_cluster_ids:
        cluster_info_item = member_cluster_infos[cluster_id]
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
        # "current_level": state["level"] + 1, #invalid update err
    }


def aggregate_renames(state: ClusterState) -> dict:
    return state


def should_continue(
    state: ClusterState,
) -> Literal["prepare_next_level", "save_final_level", "__end__"]:
    current_level = state["current_level"]
    if current_level + 1 < len(state["hierarchy"]):
        return "prepare_next_level"
    else:
        return "save_final_level"


def after_base_cluster(
    state: ClusterState,
) -> Literal["prepare_next_level", "save_final_level"]:
    """
    After base clustering, decide whether to:
    - Save and end (if only 1 level in hierarchy)
    - Move to next level for hierarchical clustering (if multiple levels)
    """
    current_level = state["current_level"]

    if len(state["hierarchy"]) == 1:
        return "save_final_level"
    else:
        return "prepare_next_level"


def save_final_level(state: ClusterState) -> dict:
    # because this doesn't go through prepare_next_level()

    # If we're at level 0 and there's only one level in hierarchy, save base clusters
    # Otherwise save parent_clusters (higher-level clusters)
    if state["current_level"] == 0 and len(state["hierarchy"]) == 1:
        final_level_clusters = {f"level_{state['current_level']}": state["clusters"]}
    else:
        final_level_clusters = {
            f"level_{state['current_level']}": state["parent_clusters"]
        }

    return {
        "all_clusters_by_level": final_level_clusters,
    }

cluster_builder = StateGraph(ClusterState, output_schema=ClusterStateOutput)
cluster_builder.add_node(base_cluster)
cluster_builder.add_node(embed)
cluster_builder.add_node(generate_neighborhoods_step)
cluster_builder.add_node(propose_clusters)
cluster_builder.add_node(dedup_clusters)
cluster_builder.add_node(assign_single_cluster)
cluster_builder.add_node(aggregate_assignments)
cluster_builder.add_node(rename_cluster_group)
cluster_builder.add_node(aggregate_renames)
cluster_builder.add_node(prepare_next_level)
cluster_builder.add_node(save_final_level)

cluster_builder.set_entry_point("base_cluster")
cluster_builder.add_conditional_edges(
    "base_cluster", after_base_cluster, ["prepare_next_level", "save_final_level"]
)
cluster_builder.add_edge("embed", "generate_neighborhoods_step")
cluster_builder.add_edge("generate_neighborhoods_step", "propose_clusters")
cluster_builder.add_edge("propose_clusters", "dedup_clusters")

# same thing as summaries for assign clusters
cluster_builder.add_conditional_edges(
    "dedup_clusters", map_assign_clusters, ["assign_single_cluster"]
)
cluster_builder.add_edge("assign_single_cluster", "aggregate_assignments")


cluster_builder.add_conditional_edges(
    "aggregate_assignments", map_rename_clusters, ["rename_cluster_group"]
)
# and same for rename clusters
cluster_builder.add_edge("rename_cluster_group", "aggregate_renames")

cluster_builder.add_conditional_edges(
    "aggregate_renames", should_continue, ["prepare_next_level", "save_final_level"]
)
cluster_builder.add_edge("prepare_next_level", "embed")
cluster_builder.add_edge("save_final_level", "__end__")
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
    all_clusters_by_level: Annotated[dict, merge_dict]

def load_examples(state: State) -> dict:
    partitions = state["partitions"]
    hierarchy = state["hierarchy"]
    dataset_name = state["dataset_name"]

    if not hierarchy:
        raise ValueError("hierarchy cannot be empty")

    if partitions is not None:
        num_partitions = len(partitions.keys())
        num_top_level_clusters = hierarchy[-1]
        if num_partitions != num_top_level_clusters:
            warnings.warn(
                f"Number of partitions ({num_partitions}) does not match number of "
                f"top-level clusters ({num_top_level_clusters})"
            )

    client = Client()
    examples = list(
        client.list_examples(
            dataset_name=dataset_name,
            limit=state["sample"] if state.get("sample") else None,
        )
    )
    total_examples = len(examples)
    validate_hierarchy(hierarchy, total_examples)  # Gives you an option to quit

    print(f"\nProcessing {total_examples} examples across the following partitions:")
    return {"total_examples": total_examples, "examples": examples}


# TODO figure out how to tqdm with langgraph
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

    print(", ".join(set(summaries_by_partition.keys())))
    print("\n[tqdm would be helpful here]\n")

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
    summary_prompt: str,  
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
        config={
            "summary_prompt": summary_prompt,
            "max_concurrency": max_concurrency,
            "recursion_limit": 100,
        },
    )
    return results


def save_langgraph_results(results, save_path=None):
    if save_path is None:
        save_path = "./clustering_results"
    os.makedirs(save_path, exist_ok=True)

    # 1. combined.csv: save examples with clustering info
    all_clusters = results.get("all_clusters_by_level", {})
    num_clusters = sum(len(level_clusters) for level_clusters in all_clusters.values())
    print(f"Clustering complete, saving results to {save_path}/ ...")
    # print(f"Total clusters created: {num_clusters}")
    examples_data = []

    examples = results.get("examples", [])
    summaries = results.get("summaries", [])

    summary_map = {summary["example_id"]: summary for summary in summaries if summary}

    # langgraph doesn't track example-to-cluster assignments directly
    # adding that would make this a lot shorter
    example_assignments = {}

    # Get all clusters from results
    all_clusters = results.get("all_clusters_by_level", {})

    # Process each level to build cluster assignments
    for level_key, level_clusters in all_clusters.items():
        level_num = int(level_key.split("_")[1])

        if level_num == 0:
            # Base level - examples are directly in the cluster
            for cluster_id, cluster_info in level_clusters.items():
                if "examples" in cluster_info:
                    for example_id in cluster_info["examples"]:
                        if example_id not in example_assignments:
                            example_assignments[example_id] = {}
                        example_assignments[example_id][level_key] = {
                            "id": str(cluster_id),
                            "name": cluster_info.get("name", ""),
                        }
        else:
            # Higher levels - need to trace back through member_clusters
            # For each member cluster, find examples that belong to it.
            for cluster_id, cluster_info in level_clusters.items():
                if "member_clusters" in cluster_info:
                    member_cluster_ids = cluster_info["member_clusters"]
                    for member_cluster_id in member_cluster_ids:
                        prev_level_key = f"level_{level_num - 1}"
                        if prev_level_key in all_clusters:
                            prev_cluster_info = all_clusters[prev_level_key].get(
                                member_cluster_id
                            )
                            if prev_cluster_info and "examples" in prev_cluster_info:
                                for example_id in prev_cluster_info["examples"]:
                                    if example_id not in example_assignments:
                                        example_assignments[example_id] = {}
                                    example_assignments[example_id][level_key] = {
                                        "id": str(cluster_id),
                                        "name": cluster_info.get("name", ""),
                                    }

    # Process each example - get base cluster info, intermediate levels, top level
    for example in examples:
        example_id = example.id
        summary = summary_map.get(example_id)

        if not summary:
            continue

        clustering = example_assignments.get(example_id, {})
        base_cluster_id = clustering.get("level_0", {}).get("id", None)
        base_cluster_name = clustering.get("level_0", {}).get("name", "")

        intermediate_clusters = {}
        for level_key in clustering.keys():
            if level_key != "level_0":
                level_num = level_key.split("_")[1]
                intermediate_clusters[f"level_{level_num}_id"] = clustering[
                    level_key
                ].get("id", None)
                intermediate_clusters[f"level_{level_num}_name"] = clustering[
                    level_key
                ].get("name", "")

        # Get top level info (highest level reached)
        max_level = (
            max([int(k.split("_")[1]) for k in clustering.keys()]) if clustering else 0
        )
        top_level_key = f"level_{max_level}"
        if top_level_key in clustering:
            top_cluster_id = clustering[top_level_key]["id"]
            top_cluster_name = clustering[top_level_key]["name"]
        elif "level_0" in clustering:
            top_cluster_id = clustering["level_0"].get("id", None)
            top_cluster_name = clustering["level_0"].get("name", "")
        else:
            top_cluster_id = None
            top_cluster_name = ""

        full_example = ""
        if hasattr(example, "inputs") and example.inputs:
            if isinstance(example.inputs, dict):
                input_parts = []
                for key, value in example.inputs.items():
                    if isinstance(value, str) and value.strip():
                        input_parts.append(f"{key}: {value}")
                full_example = "\n".join(input_parts)
            elif isinstance(example.inputs, str):
                full_example = example.inputs
        else:
            full_example = summary["summary"]  # Fallback to summary

        row_data = {
            "example_id": example_id,
            "full_example": full_example,
            "summary": summary["summary"],
            "partition": summary["partition"],
            "base_cluster_id": base_cluster_id,
            "base_cluster_name": base_cluster_name,
        }

        row_data.update(intermediate_clusters)

        row_data.update(
            {
                "top_cluster_id": top_cluster_id,
                "top_cluster_name": top_cluster_name,
            }
        )

        examples_data.append(row_data)

    examples_df = pd.DataFrame(examples_data)
    examples_df.to_csv(f"{save_path}/combined.csv", index=False)
    print(f"- Saved {len(examples_data)} examples to combined.csv")

    # 2. csv by level: Save ALL clusters from ALL levels
    all_clusters_by_level = {"level_0": [], "level_1": [], "level_2": []}

    total_clusters = 0
    for level_key, level_clusters in all_clusters.items():
        if level_key not in all_clusters_by_level:
            all_clusters_by_level[level_key] = []

        for cluster_id, cluster_data in level_clusters.items():
            total_clusters += 1

            row = {
                "cluster_id": str(cluster_id),
                "name": cluster_data.get("name", ""),
                "description": cluster_data.get("description", ""),
                "size": cluster_data.get("size", 0),
                "partition": cluster_data.get("partition", "Default"),
            }

            if "total_size" in cluster_data:
                row["total_size"] = cluster_data["total_size"]
            if "member_clusters" in cluster_data:
                member_uuids = cluster_data["member_clusters"]
                if isinstance(member_uuids, list):
                    row["member_clusters"] = str([str(uuid) for uuid in member_uuids])
                else:
                    row["member_clusters"] = str(member_uuids)

            all_clusters_by_level[level_key].append(row)

    # Save each level
    for level_name, cluster_list in all_clusters_by_level.items():
        if not cluster_list:
            continue

        df = pd.DataFrame(cluster_list)
        df = df.sort_values("size", ascending=False)
        output_path = f"{save_path}/{level_name}_clusters.csv"
        df.to_csv(output_path, index=False)
        print(f"- Saved {len(cluster_list)} clusters to {level_name}_clusters.csv")

    print("\n")
