import operator
import random
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from logging import getLogger
from typing import Literal, Sequence

import numpy as np
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from langsmith import Client
from typing_extensions import TypedDict, Annotated

from open_clio.internal.schemas import SummaryAndPartition
from open_clio.prompts import (
    ASSIGN_CLUSTER_INSTR,
    CRITERIA,
    RENAME_CLUSTER_INSTR,
)
from open_clio.generate_helpers import (
    _perform_base_clustering,
    _embed_cluster_descriptions,
    _generate_neighborhoods,
    _propose_clusters_from_neighborhoods,
    _deduplicate_proposed_clusters,
    _summarize_run,
)

DEFAULT_SUMMARIZATION_CONCURRENCY = 5

logger = getLogger(__name__)
client = Client()
embedder = init_embeddings("openai:text-embedding-3-small")
llm = init_chat_model(
    "anthropic:claude-sonnet-4-20250514",
    temperature=0.2,
    max_tokens=500,
    configurable_fields=("temperature", "max_tokens", "model"),
)


# Subgraph for clustering within partitions
# State definitions

def merge_dict(l: dict, r: dict) -> dict:
    result = l.copy()
    for key, value in r.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


class ClusterState(TypedDict):
    partition: str
    partition_description: str
    hierarchy: list[int]
    summaries_and_partitions: Annotated[list[SummaryAndPartition], lambda l, r: l + r] #changed from summaries
    total_runs: int
    # internal nodes
    current_level_clusters: Annotated[dict[int, dict], merge_dict]
    proposed_clusters: list[str] | None
    deduplicated_clusters: list[str] | None
    current_level: int
    # For assign_single_cluster
    cluster_id: int | None
    cluster_info: dict | None
    deduplicated: list[str] | None
    # For rename_cluster_group
    hl_name: str | None
    member_cluster_ids: list | None
    member_cluster_infos: dict | None
    level: int | None
    # For generate_neighborhoods
    neighborhood_labels: list[int] | None
    target_clusters: int | None
    # Results from this level needed for next
    cluster_assignments: Annotated[dict, merge_dict]
    parent_clusters: Annotated[dict, merge_dict]
    # Track all clusters by level
    all_clusters: Annotated[dict, merge_dict]

class ClusterStateOutput(TypedDict):
    all_clusters: Annotated[dict, merge_dict]

class Config(TypedDict):
    max_concurrency: int | None

# Core clustering nodes
def base_cluster(state: ClusterState) -> dict:
    curr_level = 0
    total_runs = len(state["summaries_and_partitions"])
    full_total_runs = state.get("total_runs", total_runs)

    if total_runs < full_total_runs:
        ratio = total_runs / full_total_runs
        target_base_clusters = int(round(state["hierarchy"][0] * ratio))
    else:
        target_base_clusters = state["hierarchy"][0]

    cluster_list = _perform_base_clustering(
        state["summaries_and_partitions"],
        target_base_clusters,
        state["partition"],
        state["hierarchy"],
        state.get("partition_description", ""),
    )
    current_level_clusters = {cluster["id"]: cluster for cluster in cluster_list}

    return {"current_level_clusters": current_level_clusters, "current_level": curr_level}


def embed(state: ClusterState) -> dict:
    current_clusters = state["current_level_clusters"]
    cluster_embeddings, cluster_ids = _embed_cluster_descriptions(current_clusters)

    for i, cluster_id in enumerate(cluster_ids):
        if cluster_id in current_clusters:
            current_clusters[cluster_id]["embeddings"] = cluster_embeddings[i]

    return {"current_level_clusters": current_clusters}


def generate_neighborhoods_step(state: ClusterState) -> dict:
    curr_level = state["current_level"]

    # If this is the highest level and partitions defined, skip intermediate steps
    if (
        curr_level == (len(state["hierarchy"]) - 1)
        and state.get("partition") != "Default"
    ):
        return {
            "neighborhood_labels": None,
            "target_clusters": None,
        }

    total_runs = len(state["summaries_and_partitions"])
    full_total_runs = state.get("total_runs", total_runs)

    if total_runs < full_total_runs:
        ratio = total_runs / full_total_runs
        target_clusters = int(round(state["hierarchy"][curr_level] * ratio))
    else:
        target_clusters = state["hierarchy"][curr_level]

    cluster_embeddings = []
    cluster_ids = []
    for cluster_id, cluster_info in state["current_level_clusters"].items():
        if "embeddings" in cluster_info:
            cluster_embeddings.append(cluster_info["embeddings"])
            cluster_ids.append(cluster_id)

    cluster_embeddings = np.array(cluster_embeddings)

    if len(cluster_embeddings) < 2:
        neighborhood_labels = np.array([0])  # All in same neighborhood
    else:
        neighborhood_labels, _ = _generate_neighborhoods(
            cluster_embeddings, len(cluster_embeddings), target_clusters
        )
        neighborhood_labels = np.array(neighborhood_labels)

    return {
        "neighborhood_labels": neighborhood_labels,
        "target_clusters": target_clusters,
    }


def propose_clusters(state: ClusterState) -> dict:
    # If this is the highest level and partitions defined, skip intermediate steps
    neighborhood_labels = state.get("neighborhood_labels")
    if neighborhood_labels is None:
        return {
            "proposed_clusters": None,
        }

    current_clusters = state["current_level_clusters"]
    cluster_ids = list(current_clusters.keys())
    k_neighborhoods = len(set(neighborhood_labels))
    target_clusters = state["target_clusters"]

    proposed = _propose_clusters_from_neighborhoods(
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

    # If this is the highest level and partitions defined, skip intermediate steps
    if not proposed:
        return {"deduplicated_clusters": [state.get("partition")]}

    deduplicated = _deduplicate_proposed_clusters(proposed, target_clusters)
    return {"deduplicated_clusters": deduplicated}


def map_assign_clusters(state: ClusterState) -> list[Send]:
    current_clusters = state["current_level_clusters"]  # Current level clusters
    deduplicated = state["deduplicated_clusters"]  # Parent clusters
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
    cluster_id = state["cluster_id"]
    cluster_info_item = state["cluster_info"]
    deduplicated = state["deduplicated"]

    # If this is the highest level and partitions defined, assign to partition name
    if not deduplicated:
        return {
            "cluster_assignments": {cluster_id: state["partition"]},
        }

    shuffled = deduplicated.copy()
    random.shuffle(shuffled)
    higher_level_text = "\n".join([f"<cluster>{name}</cluster>" for name in shuffled])

    prompt = ASSIGN_CLUSTER_INSTR.format(
        higher_level_text=higher_level_text,
        specific_cluster_name=cluster_info_item["name"],
        specific_cluster_description=cluster_info_item["description"],
    )

    try:
        response = llm.invoke(
            [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": """I understand. I'll evaluate the specific cluster and assign it to
the most appropriate higher-level cluster. 
<scratchpad>""",
                },
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
    current_clusters = state["current_level_clusters"]
    assignments = state["cluster_assignments"]
    level = state["current_level"]
    partition = state["partition"]
    partition_description = state["partition_description"]
    hierarchy = state["hierarchy"]

    valid_assignments = {k: v for k, v in assignments.items() if k in current_clusters}

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
                "partition_description": partition_description,
                "member_cluster_infos": {
                    cluster_id: current_clusters[cluster_id]
                    for cluster_id in member_ids
                },
                "hierarchy": hierarchy,
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
    hierarchy = state["hierarchy"]
    hl_id = uuid.uuid4()
    partition_description = state["partition_description"]

    cluster_list = []
    total_size = 0
    for cluster_id in member_cluster_ids:
        cluster_info_item = member_cluster_infos[cluster_id]
        cluster_list.append(f"<cluster>({cluster_info_item['name']})</cluster>")
        total_size += cluster_info_item.get(
            "size", cluster_info_item.get("total_size", 1)
        )

    cluster_list_text = "\n".join(cluster_list)

    # If this is the highest level in hierarchy with a non-Default partition,
    # use partition name as parent cluster name
    if level == len(hierarchy) - 1 and partition != "Default":
        parent_cluster = {
            "name": partition,
            "description": partition_description,
            "member_clusters": member_cluster_ids,
            "total_size": total_size,
            "size": len(member_cluster_ids),
            "partition": partition,
        }
        return {
            "parent_clusters": {hl_id: parent_cluster},
        }

    prompt = RENAME_CLUSTER_INSTR.format(
        cluster_list_text=cluster_list_text, criteria=CRITERIA
    )

    try:
        response = llm.invoke(
            [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": """Sure, I will provide a clear, precise, and accurate summary and
name for this cluster, which I will follow with the name: <summary>""",
                },
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
    }


def aggregate_renames(state: ClusterState) -> dict:
    return state


# Hierarchy level management

def after_base_cluster(
    state: ClusterState,
) -> Literal["prepare_next_level", "save_final_level"]:
    """
    After base clustering, decide whether to:
    - Save and end (if only 1 level in hierarchy)
    - Move to next level for hierarchical clustering (if multiple levels)
    """
    if len(state["hierarchy"]) == 1:
        return "save_final_level"
    else:
        return "prepare_next_level"


def is_next_level(
    state: ClusterState,
) -> Literal["prepare_next_level", "save_final_level", "__end__"]:
    """
    Check if we should move to the next level or save this as the final level
    """
    current_level = state["current_level"]
    if current_level + 1 < len(state["hierarchy"]):
        return "prepare_next_level"
    else:
        return "save_final_level"


def prepare_next_level(state: ClusterState) -> dict:
    """
    Saves clusters from the current level into `all_clusters` dict and resets fields that are only relevant
    to the current level.
    """
    current_level_clusters = {f"level_{state['current_level']}": state["current_level_clusters"]}
    existing_clusters = state.get("all_clusters", {})
    updated_clusters = {**existing_clusters, **current_level_clusters}

    if state["current_level"] == 0:
        next_clusters = state["current_level_clusters"]
    else:
        next_clusters = state["parent_clusters"]

    return {
        "current_level_clusters": next_clusters, 
        "current_level": state["current_level"] + 1,
        "all_clusters": updated_clusters,
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
        "partition": state["partition"],
        "partition_description": state["partition_description"],
    } #level and current level? TODO


def save_final_level(state: ClusterState) -> dict:
    """
    Save the final level clusters before ending
    """
    if state["current_level"] == 0 and len(state["hierarchy"]) == 1:
        final_level_clusters = {f"level_{state['current_level']}": state["current_level_clusters"]}
    else:
        final_level_clusters = {
            f"level_{state['current_level']}": state["parent_clusters"]
        }
    return {
        "all_clusters": final_level_clusters,
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
cluster_builder.add_conditional_edges(
    "dedup_clusters", map_assign_clusters, ["assign_single_cluster"]
)
cluster_builder.add_edge("assign_single_cluster", "aggregate_assignments")
cluster_builder.add_conditional_edges(
    "aggregate_assignments", map_rename_clusters, ["rename_cluster_group"]
)
cluster_builder.add_edge("rename_cluster_group", "aggregate_renames")
cluster_builder.add_conditional_edges(
    "aggregate_renames", is_next_level, ["prepare_next_level", "save_final_level"]
)
cluster_builder.add_edge("prepare_next_level", "embed")
cluster_builder.add_edge("save_final_level", "__end__")
cluster_graph = cluster_builder.compile()


# Main graph that orchestrates clustering by partition
class State(TypedDict):
    action: Literal["summarize", "cluster"] | None
    hierarchy: Annotated[list[int] | None, lambda l, r: l if l is not None else r]
    partitions: dict | None
    runs: Annotated[list, operator.add]
    summary_prompt: str | None
    summaries_and_partitions: Annotated[
        list[SummaryAndPartition], lambda l, r: l + r
    ]
    total_runs: Annotated[int, lambda l, r: max(l, r)]
    current_level_clusters: Annotated[dict, merge_dict]
    all_clusters: Annotated[dict, merge_dict]


class OutputState(TypedDict):
    summaries_and_partitions: Annotated[list[SummaryAndPartition], lambda l, r: l + r]
    all_clusters: Annotated[dict, merge_dict]


def map_summaries(state: State) -> list[Send]:
    return [
        Send(
            "summarize",
            {
                "run": r,
                "partitions": state.get("partitions") or {},
                "summary_prompt": state.get("summary_prompt"),
            },
        )
        for r in state["runs"]
    ]

async def summarize(state: State) -> dict:
    run = state["run"]
    summary_prompt = state.get("summary_prompt")
    if summary_prompt is None:
        summary_prompt = """Generate a concise summary of this run: {{run.inputs}} {{run.outputs}}
Guidelines:
- Focus on the specific task, problem, or domain addressed
- Highlight key technical details, methods, or outcomes when relevant
- Capture the core purpose and any notable results or decisions
- Use precise, descriptive language rather than generic phrases
- Avoid filler words like "User requested," "I provided," or "Discussion about"
- Aim for 8-15 words that would help someone quickly identify this run later"""

    summary_and_partition = await _summarize_run(
        state["partitions"],
        run,
        summary_prompt,
    )
    # Don't error on run summary failures, aggregate_summaries has a tolerance threshold
    if summary_and_partition and summary_and_partition.get("run_id") is not None:
        return {"summaries_and_partitions": [summary_and_partition]}


def aggregate_summaries(state: State) -> dict:
    summaries_and_partitions = state.get("summaries_and_partitions")
    total = len(summaries_and_partitions)
    # Remove failed summaries 
    for result in summaries_and_partitions:
        if not result["summary"]:
            summaries_and_partitions.remove(result)
    # Fails if > 25% of summaries fail
    if (total - len(summaries_and_partitions)) / total > 0.25:
        raise ValueError(f"Too many summaries failed: {total - len(summaries_and_partitions)}/{total}")
    return {}


def map_partitions(state: State) -> list[Send]:
    summaries_by_partition = defaultdict(list)
    for result in state["summaries_and_partitions"]:
        if result["summary"]:
            summaries_by_partition[result["partition"]].append(result)

    sends = []
    partitions = state.get("partitions", {})
    for partition, cat_summaries in summaries_by_partition.items():
        run_ids = [s["run_id"] for s in cat_summaries]
        partition_runs = []
        for r in state["runs"]:
            if isinstance(r, dict) and "id" in r:
                run_id = r["id"]
            elif isinstance(r, dict) and "metadata" in r and "run_id" in r["metadata"]:
                run_id = r["metadata"]["run_id"]
            else:
                raise ValueError(f"Run {r} has no ID")
            if run_id in run_ids:
                partition_runs.append(r)
        partition_description = (
            partitions.get(partition, "") if state.get("partitions") else None
        )

        sends.append(
            Send(
                "cluster_partition",
                {
                    "partition": partition,
                    "partition_description": partition_description,
                    "runs": partition_runs,
                    "summaries_and_partitions": cat_summaries,
                    "hierarchy": state["hierarchy"],
                    "total_runs": state["total_runs"],
                },
            )
        )
    return sends


def route_action(state: State) -> list[Send]:
    """Route to either summarize path or cluster_partition path based on action."""
    if state.get("action") == "summarize":
        return map_summaries(state)
    elif state.get("action") == "cluster":
        return map_partitions(state)
    else:
        raise ValueError(f"Invalid action: {state.get('action')}")


partitioned_cluster_builder = StateGraph(State)
partitioned_cluster_builder.add_node("summarize", summarize)
partitioned_cluster_builder.add_node("cluster_partition", cluster_graph)
partitioned_cluster_builder.add_node(aggregate_summaries)
partitioned_cluster_builder.add_conditional_edges(
    START,
    route_action,
)
partitioned_cluster_builder.add_edge("summarize", "aggregate_summaries")
partitioned_cluster_builder.add_edge("aggregate_summaries", END)
partitioned_cluster_builder.add_edge("cluster_partition", END)
partitioned_cluster_graph = partitioned_cluster_builder.compile().with_config(
    {"max_concurrency": 5}
)
