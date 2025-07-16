import asyncio
import os
import random
import time
from collections import defaultdict
from typing import Sequence
import warnings
import uuid
from tqdm import tqdm

from open_clio.internal.utils import gated_coro
from open_clio.internal import schemas

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
import numpy as np
import pandas as pd
from langsmith import Client
from langsmith import schemas as ls_schemas
from open_clio.prompts import (
    ASSIGN_CLUSTER_INSTR,
    CRITERIA,
    DEDUPLICATE_CLUSTERS_INSTR,
    NAME_CLUSTER_INSTR,
    PROPOSE_CLUSTERS_INSTR,
    RENAME_CLUSTER_INSTR,
    SUMMARIZE_INSTR,
)
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans
from logging import getLogger

logger = getLogger(__file__)

DEFAULT_SUMMARIZATION_CONCURRENCY = 5

client = Client()
embedder = init_embeddings("openai:text-embedding-3-small")
llm = init_chat_model(
    "anthropic:claude-sonnet-4-20250514",
    temperature=0.2,
    max_tokens=500,
    configurable_fields=("temperature", "max_tokens", "model"),
)


async def summarize_example(
    example: ls_schemas.Example, partitions: dict[str, str], summary_prompt: str
) -> schemas.ExampleSummary | None:
    """Use an LLM to generate a summary for a single example."""

    class ResponseFormatter(BaseModel):
        summary: str = Field(
            description="A structured summary of the support conversation that captures the main task, request, or purpose. Focus on what the user is asking for, be specific about the subject matter or domain, and include context about the purpose or use case when relevant. Do NOT include phrases like 'User requested' or 'I understand' - start directly with the action/task."
        )
        partition: str = Field(
            description=f"The main product partition this support request belongs to. Must be one of: {list(partitions.keys()) if partitions else ['Default']}"
        )

    # Create structured LLM here
    structured_llm = llm.with_structured_output(ResponseFormatter)

    conversation_text = str(example.inputs)

    # If no partitions provided, all in same partition
    if not partitions:
        partitions_str = (
            "- Default: All items in the dataset belong to this partition by default"
        )
    else:
        partitions_str = "\n".join(f"- {k}: {v}" for k, v in partitions.items())

    summary_prompt_w_partitions = SUMMARIZE_INSTR.format(
        summary_prompt=summary_prompt, partitions=partitions_str
    )

    messages = [
        {
            "role": "user",
            "content": f"The following is a conversation between an AI assistant and a user:\n\n{conversation_text}",
        },
        {
            "role": "assistant",
            "content": "I understand.",
        },
        {
            "role": "user",
            "content": f"{summary_prompt_w_partitions}",
        },
        {
            "role": "assistant",
            "content": "Sure, I'll analyze this conversation and provide a structured summary: <answer>",
        },
    ]

    try:
        response = await structured_llm.ainvoke(messages)

        res = response.summary
        partition = response.partition

    except Exception as e:
        logger.error(f"Error processing example {example.id}: {e}")
        try:
            raw_response = await llm.ainvoke(messages)
            logger.error(
                f"Raw LLM response (before parsing) for example {example.id}: {raw_response}"
            )
            # print(
            #    f"Raw LLM response (before parsing) for example {example.id}: {raw_response}"
            #)
        except Exception as raw_e:
            logger.error(
                f"Could not get raw response for example {example.id}: {raw_e}"
            )
        return None

    return {"summary": res, "partition": partition, "example_id": example.id}


async def summarize_all(
    examples: list,
    partitions: dict[str, str],
    summary_prompt: str,
    *,
    max_concurrency: int,
) -> list[schemas.ExampleSummary | None]:
    """Generate summaries for all examples in the dataset."""
    logger.info("Generating example summaries")
    print("Generating summaries...")

    semaphore = asyncio.Semaphore(max_concurrency)

    tasks = [
        gated_coro(summarize_example(example, partitions, summary_prompt), semaphore)
        for example in examples
    ]
    summaries = []
    with tqdm(total=len(tasks), desc="Generating summaries") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            summaries.append(result)
            pbar.update(1)

    # summaries = await asyncio.gather(*tasks)
    num_successful = sum(s is not None for s in summaries)
    total_ex = len(examples)
    logger.info(
        f"Summaries successfully generated for {num_successful} out of {total_ex} examples"
    )
    print(f"Generated {num_successful}/{total_ex} summaries")
    return summaries


def perform_base_clustering(
    summaries, partition_k, partition
) -> list[schemas.ClusterInfo]:
    """
    Perform the initial base clustering for a partition.

    Args:
        summaries: List of summaries to cluster
        partition_k: Number of clusters to create
        id_offset: Offset for cluster IDs
        partition: partition name for logging

    Returns:
        tuple: (cluster_info, cluster_labels)
    """
    # Safety check: partition_k must be at least 1
    if partition_k < 1:
        partition_k = 1
    # generate embeddings
    embeddings = np.array(embedder.embed_documents([s["summary"] for s in summaries]))

    # apply kmeans clustering
    kmeans = KMeans(n_clusters=partition_k, random_state=42, n_init=10, max_iter=300)
    clusters = kmeans.fit_predict(embeddings)
    # if len(np.unique(clusters)) >= 2:
    #    silhouette = silhouette_score(embeddings, clusters)
    #    print(f"silhoutte score: {silhouette}")

    # generate descriptions for all clusters
    return generate_cluster_descriptions(clusters, summaries, embeddings, partition)


def deduplicate_base_clusters(cluster_info, partition_ktop):
    pass


def embed_cluster_descriptions(current_clusters):
    """
    Step 1: Embed cluster descriptions for hierarchical clustering.

    Args:
        current_clusters: Dictionary of current cluster information

    Returns:
        tuple: (cluster_embeddings, cluster_ids)
    """
    cluster_names_descriptions = []
    cluster_ids = []

    for cluster_id, info in current_clusters.items():
        text = f"{info['name']}: {info['description']}"
        cluster_names_descriptions.append(text)
        cluster_ids.append(cluster_id)

    logger.info(f"Embedding {len(cluster_ids)} cluster descriptions...")
    cluster_embeddings = np.array(embedder.embed_documents(cluster_names_descriptions))
    time.sleep(1.0)  # Sleep after embedding step

    return cluster_embeddings, cluster_ids


def generate_neighborhoods(cluster_embeddings, num_clusters, target_clusters):
    """
    Step 2: Generate neighborhoods using k-means clustering.

    Args:
        cluster_embeddings: Embeddings of cluster descriptions
        num_clusters: Number of current clusters

    Returns:
        tuple: (neighborhood_labels, k_neighborhoods)
    """
    if target_clusters and num_clusters > target_clusters * 2:
        # big reduction, use fewer neighborhoods
        k_neighborhoods = max(3, target_clusters)
    else:
        # small reduction, bigger neighborhoods
        k_neighborhoods = min(6, num_clusters // 2)

    k_neighborhoods = max(
        2, min(k_neighborhoods, num_clusters)
    )  # At least 2 neighborhoods

    logger.info(f"Proposing {k_neighborhoods} higher level clusters...")
    kmeans_nbh = KMeans(
        n_clusters=k_neighborhoods, random_state=42, n_init=10, max_iter=300
    )
    neighborhood_labels = kmeans_nbh.fit_predict(cluster_embeddings)

    return neighborhood_labels, k_neighborhoods


def propose_clusters_from_neighborhoods(
    current_clusters, cluster_ids, neighborhood_labels, k_neighborhoods, target_clusters
):
    """
    Step 3: Propose new clusters for each neighborhood.

    Args:
        current_clusters: Dictionary of current cluster information
        cluster_ids: List of cluster IDs
        neighborhood_labels: Labels indicating which neighborhood each cluster belongs to
        k_neighborhoods: Number of neighborhoods
        target_clusters: Target number of clusters for this level

    Returns:
        list: Proposed cluster names
    """
    proposed = []
    clusters_per_neighborhood = max(1, target_clusters // k_neighborhoods)

    for neighborhood_id in range(k_neighborhoods):
        neighborhood_mask = neighborhood_labels == neighborhood_id
        neighborhood_cluster_ids = [
            cluster_ids[i] for i in range(len(cluster_ids)) if neighborhood_mask[i]
        ]
        neighborhood_clusters = {
            cid: current_clusters[cid] for cid in neighborhood_cluster_ids
        }

        # build cluster list for prompt
        cluster_list = []
        for cluster_id, info in neighborhood_clusters.items():
            cluster_list.append(
                f"<cluster>{info['name']}: {info['description']}</cluster>"
            )

        cluster_list_text = "\n".join(cluster_list)

        proposing_user_prompt = PROPOSE_CLUSTERS_INSTR.format(
            cluster_list_text=cluster_list_text,
            clusters_per_neighborhood=clusters_per_neighborhood,
            criteria=CRITERIA,
            min_cpn=max(1, int(0.5 * clusters_per_neighborhood)),
            max_cpn=max(2, int(1.5 * clusters_per_neighborhood)),
        )
        proposing_assistant_prompt = """ I understand. I'll evaluate the clusters and provide higher-level
cluster names that could encompass multiple sub-clusters within the LangChain ecosystem. 
<scratchpad>"""

        try:
            response = llm.invoke(
                [
                    {"role": "user", "content": proposing_user_prompt},
                    {"role": "assistant", "content": proposing_assistant_prompt},
                ],
                max_tokens=1000,
                temperature=1.0,
            )

            proposed_names = []
            content = response.content
            # extract answer section
            if "<answer>" in content and "</answer>" in content:
                ans_start = content.find("<answer>")
                ans_end = content.find("</answer>")
                ans_text = content[ans_start:ans_end].strip()

                for line in ans_text.split("\n"):
                    line = line.strip()
                    if line and (
                        line[0].isdigit() or line.startswith("-")
                    ):  # not sure if numbering or bullet
                        name = (
                            line.split(".", 1)[1].strip()
                            if "." in line
                            else line.strip()
                        )
                        if name.startswith("[") and name.endswith("]"):
                            name = name[1:-1]
                        proposed_names.append(name)
            if not proposed_names:
                # Create fallback names based on the actual clusters in this neighborhood
                if len(neighborhood_clusters) > 0:
                    sample_names = [
                        info["name"]
                        for info in list(neighborhood_clusters.values())[:2]
                    ]
                    proposed_names = [
                        f"Handle {' and '.join(sample_names[:2])} related requests"
                    ]
                else:
                    proposed_names = [f"Cluster Group {neighborhood_id}"]

            proposed.extend(proposed_names)
            time.sleep(1.0)

        except Exception as e:
            logger.error(
                f"Error proposing clusters for neighborhood {neighborhood_id}: {e}"
            )
            time.sleep(1.0)  # Sleep even on error to respect rate limits
    logger.info(f"Proposed clusters: {proposed}")
    return proposed


def deduplicate_proposed_clusters(proposed, target_clusters):
    """
    Step 4: Deduplicate proposed clusters across neighborhoods.

    Args:
        proposed: List of proposed cluster names
        target_clusters: Target number of clusters

    Returns:
        list: Deduplicated cluster names
    """
    logger.info(f"Deduplicating {len(proposed)} proposed clusters")
    if len(proposed) == 0:
        logger.error("ERROR: No clusters were proposed!")
        return []
    if len(proposed) <= target_clusters:
        return proposed

    cluster_text = "\n".join([f"<cluster>{name}</cluster>" for name in proposed])
    deduplicating_user_prompt = DEDUPLICATE_CLUSTERS_INSTR.format(
        cluster_text=cluster_text,
        target_clusters=target_clusters,
        clusters_per_neighborhood=max(
            1, target_clusters // 2
        ),  # Approximate, but at least 1
        criteria=CRITERIA,
        min_cpn=max(1, int(0.5 * target_clusters)),
        max_cpn=max(2, int(1.5 * target_clusters)),
    )

    deduplicating_assistant_prompt = f"""
I understand. I'll deduplicate the cluster names into approximately {target_clusters} names.
<scratchpad>"""

    try:
        response = llm.invoke(
            [
                {"role": "user", "content": deduplicating_user_prompt},
                {
                    "role": "assistant",
                    "content": deduplicating_assistant_prompt,
                },
            ],
            max_tokens=1000,
            temperature=1.0,
        )
        content = str(response.content)
        deduplicated = []
        if "<answer>" in content and "</answer>" in content:
            ans_start = content.find("<answer>") + 8
            ans_end = content.find("</answer>")
            ans_text = content[ans_start:ans_end].strip()

            for line in ans_text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    name = (
                        line.split(".", 1)[1].strip() if "." in line else line.strip()
                    )
                    if name.startswith("[") and name.endswith("]"):
                        name = name[1:-1]
                    deduplicated.append(name)
        else:
            logger.warning("Warning: Could not parse deduplicated clusters")
            deduplicated = proposed[:target_clusters]

    except Exception as e:
        logger.error(f"Error deduplicating clusters: {e}")
        deduplicated = proposed[:target_clusters]

    logger.info(f"Final deduplicated clusters: {deduplicated}")
    time.sleep(1.0)
    return deduplicated


def assign_clusters_to_higher_level(current_clusters, deduplicated):
    """
    Step 5: Assign clusters to higher level clusters.

    Args:
        current_clusters: Dictionary of current cluster information
        deduplicated: List of deduplicated higher-level cluster names

    Returns:
        dict: Mapping of cluster_id to assigned higher-level cluster name
    """
    logger.info(
        f"Assigning {len(current_clusters)} clusters to {len(deduplicated)} higher-level clusters..."
    )
    assignments = {}
    shuffled = deduplicated.copy()
    random.shuffle(shuffled)
    higher_level_text = "\n".join([f"<cluster>{name}</cluster>" for name in shuffled])

    for cluster_id, cluster_info_item in current_clusters.items():
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
                max_tokens=500,
                temperature=1.0,
            )
            content = str(response.content)
            if "<answer>" in content and "</answer>" in content:
                ans_start = content.find("<answer>") + 8
                ans_end = content.find("</answer>")
                assigned_cluster = content[ans_start:ans_end].strip()

                if assigned_cluster in deduplicated:
                    assignments[cluster_id] = assigned_cluster
                else:
                    best_match = None
                    for hl_name in deduplicated:
                        if (
                            assigned_cluster.lower() in hl_name.lower()
                            or hl_name.lower() in assigned_cluster.lower()
                        ):
                            best_match = hl_name
                            break
                    assignments[cluster_id] = best_match or deduplicated[0]
            else:
                assignments[cluster_id] = deduplicated[0]
        except Exception as e:
            logger.error(f"Error assigning cluster {cluster_id}: {e}")
            assignments[cluster_id] = deduplicated[0]

        time.sleep(1.0)  # Sleep after each cluster assignment

    return assignments


def rename_higher_level_clusters(current_clusters, assignments, level, partition):
    """
    Step 6: Rename higher level clusters based on assignments.

    Args:
        current_clusters: Dictionary of current cluster information
        assignments: Mapping of cluster_id to assigned higher-level cluster name
        deduplicated: List of deduplicated higher-level cluster names
        level: Current hierarchy level
        partition: partition name

    Returns:
        dict: New level clusters with names and descriptions
    """
    logger.info("Renaming higher-level clusters based on assignments...")
    new_lvl_clusters = {}

    # group clusters by their assigned HL cluster
    cluster_groups = {}
    for cluster_id, assigned_hl in assignments.items():
        if assigned_hl not in cluster_groups:
            cluster_groups[assigned_hl] = []
        cluster_groups[assigned_hl].append(cluster_id)

    for hl_name, member_cluster_ids in cluster_groups.items():
        hl_id = uuid.uuid4()
        # building list of member clusters for prompt
        cluster_list = []  # changed from members
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
                max_tokens=500,
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
                else f"Level {level} Cluster {hl_id}\n"
            )

        except Exception as e:
            logger.error(f"Error renaming cluster {hl_name}: {e}")
            name = f"Level {level} Cluster {hl_id}"
            summary = "Summary generation failed"

        new_lvl_clusters[hl_id] = {
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
        time.sleep(1.0)

    return new_lvl_clusters


def generate_cluster_descriptions(
    clusters, summaries, embeddings, partition
) -> list[schemas.ClusterInfo]:
    cluster_info = []
    num_clusters = max(clusters) + 1

    for idx in range(num_clusters):
        cluster_mask = clusters == idx
        cluster_summary_infos = [
            summary for summary, mask in zip(summaries, cluster_mask) if mask
        ]
        cluster_summaries = [s["summary"] for s in cluster_summary_infos]

        # get contrastive examples
        contrastive_summaries = get_contrastive_summaries(
            cluster_mask, embeddings, summaries
        )

        # Generate cluster ID first
        cluster_id = uuid.uuid4()

        # use them to generate the description for this cluster
        name, description = generate_single_cluster_description(
            cluster_summaries, contrastive_summaries, cluster_id
        )
        cluster_info.append(
            {
                "name": name,
                "description": description,
                "size": len(cluster_summaries),
                "summaries": [s["summary"] for s in cluster_summary_infos],
                "examples": [s["example_id"] for s in cluster_summary_infos],
                "partition": partition,
                "id": cluster_id,
            }
        )

        logger.info(
            f"Level 0 Cluster {cluster_id}: {description} ({len(cluster_summaries)} items)"
        )

    return cluster_info


def get_contrastive_summaries(cluster_mask, embeddings, summaries):
    """
    Use up to 50 examples nearest to but outside of this cluster to explain what differentiates it from other clusters.
    """
    # get contrastive ex (still within this partition)
    cluster_embeddings = embeddings[cluster_mask]
    cluster_centroid = np.mean(cluster_embeddings, axis=0)

    # get distances from centroid to all non-cluster points within partition
    non_cluster_mask = ~cluster_mask
    non_cluster_embeddings = embeddings[non_cluster_mask]
    non_cluster_summaries = [
        summaries[i] for i in range(len(summaries)) if not cluster_mask[i]
    ]

    if len(non_cluster_summaries) > 0:
        # calculate distances to centroid
        distances = np.linalg.norm(non_cluster_embeddings - cluster_centroid, axis=1)

        # get closest non-cluster summaries
        n_contrastive = min(50, len(non_cluster_summaries))
        nearest_indices = np.argsort(distances)[:n_contrastive]
        contrastive_summaries = [
            non_cluster_summaries[i]["summary"] for i in nearest_indices
        ]
    else:
        contrastive_summaries = []

    return contrastive_summaries


def generate_single_cluster_description(
    cluster_summaries, contrastive_summaries, cluster_id
):
    max_summaries = 15
    if len(cluster_summaries) > max_summaries:
        cluster_sample = np.random.choice(
            cluster_summaries, max_summaries, replace=False
        ).tolist()
    else:
        cluster_sample = cluster_summaries

    if len(contrastive_summaries) > max_summaries:
        contrastive_sample = np.random.choice(
            contrastive_summaries, max_summaries, replace=False
        ).tolist()
    else:
        contrastive_sample = contrastive_summaries

    cluster_sample = "\n".join(cluster_sample)  # list to str
    contrastive_sample = "\n".join(contrastive_sample)

    prompt = NAME_CLUSTER_INSTR.format(
        cluster_sample=cluster_sample,
        contrastive_sample=contrastive_sample,
        criteria=CRITERIA,
    )

    try:
        response = llm.invoke(
            [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": """Sure, I will provide a clear, precise, and accurate summary and name for
    this cluster. I will be descriptive and assume neither good nor bad faith. Here
    is the summary, which I will follow with the name: <summary>""",
                },
            ],
            max_tokens=500,
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
        name = content[name_start:name_end].strip()

    except Exception as e:
        print(f"Error: {e}")
        name = f"Cluster {cluster_id}"
        summary = "Error generating description"

    return name, summary


def cluster_partition_examples(
    partition: str,
    examples: list[ls_schemas.Example],
    summaries: list[schemas.ExampleSummary],
    total_examples: int,
    hierarchy: list[int],
):
    """
    Orchestrates hierarchical clustering for each user-defined partition/partition.
    Defaults to processing all examples if no partitions/partitions provided.
    """
    single_partition = len(examples) == total_examples

    if single_partition:
        partition_k = hierarchy[0]
        partition_ktop = hierarchy[-1]
        if len(hierarchy) == 2:
            scaled_level_sizes = [partition_ktop]
        else:
            scaled_level_sizes = hierarchy[1:-1]  # fix = getting index out of rnage
    else:
        partition_ratio = len(examples) / total_examples
        partition_k = round(hierarchy[0] * partition_ratio)
        partition_ktop = round(hierarchy[-1] * partition_ratio)
        scaled_level_sizes = [k * partition_ratio for k in hierarchy[1:-1]]

    logger.info(
        f"Partition '{partition}': {len(examples)} examples, target base clusters: {partition_k}, target top clusters: {partition_ktop}"
    )
    if len(hierarchy) == 1:
        print(
            f"Partition '{partition}': {len(examples)} examples → {partition_k} base clusters"
        )
    else:
        print(
            f"Partition '{partition}': {len(examples)} examples → {partition_k} base clusters → {partition_ktop} top clusters"
        )

    # perform base clustering
    logger.info("Generating base clusters...")
    if partition_k > 1:
        cluster_info = perform_base_clustering(summaries, partition_k, partition)
    else:
        # TODO
        cluster_info = [
            {
                "name": f"all {partition} examples",
                "description": f"all unclustered {partition} examples",
                "id": uuid.uuid4(),
                "size": len(summaries),
                "summaries": [s["summary"] for s in summaries],
                "examples": [s["example_id"] for s in summaries],
                "partition": partition,
            }
        ]
    # deduplicate_base_clusters(cluster_info, partition_ktop) at some point

    logger.info("Building the next level of clusters...")

    # previously index based, now based on uuid
    cluster_info_dict = {cluster["id"]: cluster for cluster in cluster_info}
    partition_hierarchy = {"level_0": cluster_info_dict, "max_level": 0}
    current_clusters = cluster_info_dict

    n_base = len(current_clusters)

    # Track example assignments at each level
    example_assignments = {
        "level_0": {eid: c["id"] for c in cluster_info for eid in c["examples"]}
    }
    levels = len(hierarchy)
    # Always use geometric progression for consistent behavior
    if levels == 1:
        # Single level hierarchy - no higher levels to build
        scaled_level_sizes = []
    elif levels == 2:
        scaled_level_sizes = [partition_ktop]
    else:
        ratio = (partition_ktop / n_base) ** (1 / (levels - 1))
        scaled_level_sizes = []
        for level in range(1, levels):
            n_level = int(n_base * (ratio**level))
            scaled_level_sizes.append(max(2, n_level))  # each level has at least 2
        scaled_level_sizes.append(partition_ktop)

    logger.info(
        f"Planned hierarchy sizes for partition '{partition}': {n_base} + {scaled_level_sizes}"
    )

    # Build clusters for each level in the hierarchy
    for level in range(1, levels):
        logger.info(f"=== STARTING LEVEL {level} ===")
        logger.info(f"Targeting {scaled_level_sizes[level - 1]} clusters")
        logger.info(f"Current clusters: {len(current_clusters)}")

        try:
            if len(current_clusters) <= scaled_level_sizes[level - 1]:
                logger.info(
                    f"Stopping at level {level - 1} (only {len(current_clusters)} clusters left)"
                )
                print(
                    f"Stopping at level {level - 1} (only {len(current_clusters)} clusters left)"
                )
                break

            # 1) embed clusters
            cluster_embeddings, cluster_ids = embed_cluster_descriptions(
                current_clusters
            )

            # 2) set target clusters for this level
            target_clusters = scaled_level_sizes[level - 1]

            # 3) generate neighbourhoods using k-means
            neighborhood_labels, k_neighborhoods = generate_neighborhoods(
                cluster_embeddings, len(current_clusters), target_clusters
            )

            # 4) propose new clusters for each neighborhood
            proposed = propose_clusters_from_neighborhoods(
                current_clusters,
                cluster_ids,
                neighborhood_labels,
                k_neighborhoods,
                target_clusters,
            )

            # 4) deduplicate across neighborhoods
            deduplicated = deduplicate_proposed_clusters(proposed, target_clusters)

            # 5) assign clusters to higher level clusters
            assignments = assign_clusters_to_higher_level(
                current_clusters, deduplicated
            )

            # 6) rename higher level clusters based on assignments
            new_lvl_clusters = rename_higher_level_clusters(
                current_clusters, assignments, level, partition
            )

            # Track example assignments for this level
            example_assignments[f"level_{level}"] = {}
            if level == 1:
                # For level 1, map from level_0 clusters to level_1 clusters
                for example_id, base_cluster_id in example_assignments[
                    "level_0"
                ].items():
                    if base_cluster_id in assignments:
                        # Find the cluster ID for this higher-level cluster name
                        # assignments maps cluster_id to proposed cluster name
                        # We need to find which new cluster this base cluster was assigned to
                        for hl_cluster_id, hl_cluster_info in new_lvl_clusters.items():
                            if base_cluster_id in hl_cluster_info.get(
                                "member_clusters", []
                            ):
                                example_assignments[f"level_{level}"][example_id] = (
                                    hl_cluster_id
                                )
                                break
            else:
                # For level 2+, map from previous level clusters to current level clusters
                previous_level = f"level_{level - 1}"
                for example_id, prev_cluster_id in example_assignments[
                    previous_level
                ].items():
                    if prev_cluster_id in assignments:
                        # Find the cluster ID for this higher-level cluster name
                        # assignments maps cluster_id to proposed cluster name
                        # We need to find which new cluster this previous cluster was assigned to
                        for hl_cluster_id, hl_cluster_info in new_lvl_clusters.items():
                            if prev_cluster_id in hl_cluster_info.get(
                                "member_clusters", []
                            ):
                                example_assignments[f"level_{level}"][example_id] = (
                                    hl_cluster_id
                                )
                                break

            partition_hierarchy[f"level_{level}"] = new_lvl_clusters
            partition_hierarchy["max_level"] = level
            current_clusters = new_lvl_clusters

            logger.info(
                f"Level {level} complete, checking if more levels are needed..."
            )

        except Exception as e:
            logger.error(f"ERROR at level {level}: {e}")
            print(f"ERROR at level {level}: {e}")
            import traceback

            traceback.print_exc()
            break

    logger.info(f"Hierarchical clustering complete for partition '{partition}'!")

    partition_updates = []
    for example in examples:
        # find the corresponding summary
        example_summary = None
        for summary in summaries:
            if summary["example_id"] == example.id:
                example_summary = summary
                break

        if example_summary is None:
            logger.warning(f"No summary found for example {example.id}")
            print(f"No summary found for example {example.id}")
            continue  # skip examples that didn't work

        clustering = {}
        for level_key, level_assignments in example_assignments.items():
            if example.id in level_assignments:
                cluster_id = level_assignments[example.id]  # not example_id anymore
                # Find cluster info for this level
                if level_key == "level_0":
                    cluster_info_for_level = cluster_info_dict
                else:
                    cluster_info_for_level = partition_hierarchy[level_key]

                if cluster_id in cluster_info_for_level:
                    clustering[level_key] = {
                        "id": str(cluster_id),  # for csv
                        "name": cluster_info_for_level[cluster_id]["name"],
                    }

        update = {
            "id": example.id,  # was example_id
            "metadata": example.metadata,
            "inputs": example.inputs,
            "outputs": {
                "summary": example_summary["summary"],
                "partition": example_summary["partition"],
                "clustering": clustering,
            },
        }
        partition_updates.append(update)

    # calculate next cluster id offset
    return partition_updates, partition_hierarchy


def save_results(all_updates, combined_hierarchy, save_path=None):
    # print results summary
    if save_path is None:
        save_path = "./clustering_results"
    logger.info("Overview of clustering results:")
    print("\nClustering results:")

    for partition, hierarchy in combined_hierarchy["partitions"].items():
        logger.info(f"Partition: {partition}")
        logger.info(f"Base clusters: {len(hierarchy['level_0'])}")
        print(f"  {partition}: {len(hierarchy['level_0'])} base clusters", end="")
        if hierarchy["max_level"] > 0:
            for level in range(1, hierarchy["max_level"] + 1):
                logger.info(
                    f"Level {level} clusters: {len(hierarchy[f'level_{level}'])}"
                )
                print(f" → {len(hierarchy[f'level_{level}'])} level {level}", end="")
        print()

    # update dataset with all cluster assignments
    logger.info("Updating the dataset with clustering results...")
    # client.update_examples(dataset_name=dataset_name, updates=all_updates)

    os.makedirs(save_path, exist_ok=True)

    # 1. combined.csv: save combined examples with full hierarchical clustering info
    logger.info(f"Saving results to {save_path}...")
    print(f"\nSaving results to {save_path}...")
    examples_data = []

    for update in all_updates:
        clustering = update["outputs"]["clustering"]
        # print(f"clustering: {clustering}")

        # Get base cluster info
        base_cluster_id = clustering.get("level_0", {}).get("id", None)
        base_cluster_name = clustering.get("level_0", {}).get("name", "")

        # Get intermediate cluster levels (level_1, level_2, etc.)
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
        if "level_1" in clustering:
            top_cluster_id = clustering["level_1"]["id"]
            top_cluster_name = clustering["level_1"]["name"]
        elif "level_0" in clustering:
            top_cluster_id = clustering["level_0"].get("id", None)
            top_cluster_name = clustering["level_0"].get("name", "")
        else:
            top_cluster_id = None
            top_cluster_name = ""

        # Create full example text (combine inputs if available)
        full_example = ""
        if "inputs" in update and update["inputs"]:
            if isinstance(update["inputs"], dict):
                # If inputs is a dict, try to extract meaningful text
                input_parts = []
                for key, value in update["inputs"].items():
                    if isinstance(value, str) and value.strip():
                        input_parts.append(f"{key}: {value}")
                full_example = "\n".join(input_parts)
            elif isinstance(update["inputs"], str):
                full_example = update["inputs"]
        else:
            full_example = update["outputs"]["summary"]  # Fallback to summary

        row_data = {
            "example_id": update["id"],
            "full_example": full_example,
            "summary": update["outputs"]["summary"],
            "partition": update["outputs"]["partition"],
            "base_cluster_id": base_cluster_id,
            "base_cluster_name": base_cluster_name,
        }

        # Add intermediate cluster levels
        row_data.update(intermediate_clusters)

        # Add top cluster info
        row_data.update(
            {
                "top_cluster_id": top_cluster_id,
                "top_cluster_name": top_cluster_name,
            }
        )
        # print(f"\nrow_data: {row_data}")

        examples_data.append(row_data)  # examples_data has row_data for every example
    examples_df = pd.DataFrame(examples_data)

    examples_df.to_csv(f"{save_path}/combined.csv", index=False)
    # looks like: example_id,full_example,summary,partition,base_cluster_id,base_cluster_name, [intermediates], top_cluster_id,top_cluster_name
    logger.info(f"Saved {len(examples_data)} examples to combined.csv")
    print(f"Saved {len(examples_data)} examples to combined.csv")

    # 2. csv by level: Save ALL clusters from ALL levels across ALL partitions
    all_clusters = {"level_0": [], "level_1": [], "level_2": []}

    total_clusters = 0
    for partition, cat_hierarchy in combined_hierarchy["partitions"].items():
        for level in range(cat_hierarchy["max_level"] + 1):
            level_key = f"level_{level}"
            if level_key in cat_hierarchy:
                for cluster_id, cluster_data in cat_hierarchy[level_key].items():
                    total_clusters += 1

                    row = {
                        "cluster_id": str(cluster_id),
                        "name": cluster_data.get("name", ""),
                        "description": cluster_data.get("description", ""),
                        "size": cluster_data.get("size", 0),
                        "partition": partition,
                    }

                    # Add level-specific fields
                    if "total_size" in cluster_data:
                        row["total_size"] = cluster_data["total_size"]
                    if "member_clusters" in cluster_data:
                        member_uuids = cluster_data["member_clusters"]
                        if isinstance(member_uuids, list):
                            row["member_clusters"] = str(
                                [str(uuid) for uuid in member_uuids]
                            )
                        else:
                            row["member_clusters"] = str(member_uuids)

                    all_clusters[level_key].append(row)

    # Save each level
    for level_name, cluster_list in all_clusters.items():
        if not cluster_list:
            continue

        df = pd.DataFrame(cluster_list)
        df = df.sort_values("size", ascending=False)
        output_path = f"{save_path}/{level_name}_clusters.csv"  # debug
        df.to_csv(output_path, index=False)  # debug
        logger.info(f"Saved {len(cluster_list)} clusters to {level_name}_clusters.csv")
        print(f"Saved {len(cluster_list)} clusters to {level_name}_clusters.csv")

    logger.info(f"See {save_path} for csv files.")
    print(f"Results saved to {save_path}/")


def validate_hierarchy(hierarchy: Sequence[int], n_examples: int) -> None:
    """Check if hierarchy makes logical sense"""
    if len(hierarchy) > 3:
        warnings.warn(
            f"Warning: {len(hierarchy)} levels may be too many for {n_examples} examples."
            f" Consider starting with <=3 levels for meaningful results."
        )

    if hierarchy[0] > n_examples:
        raise ValueError(
            f"Cannot specify more base clusters ({hierarchy[0]}) than"
            " there are dataset examples ({n_examples})."
        )
    suggested_max_k = max(int(np.sqrt(n_examples)), n_examples // 3)
    if hierarchy[0] > suggested_max_k:
        warnings.warn(
            f"Warning: {hierarchy[0]} base clusters may be too many for {n_examples} examples."
            f" Consider starting with <={suggested_max_k} clusters."
        )

    # decreasing numbers
    for i in range(len(hierarchy) - 1):
        if hierarchy[i] <= hierarchy[i + 1]:
            raise ValueError(
                f"Level {i} has {hierarchy[i]} clusters, level {i + 1} has {hierarchy[i + 1]}"
            )


async def generate_clusters(
    dataset_name: str,
    hierarchy: list,
    summary_prompt: str,  # TODO
    *,
    save_path: str | None = None,
    partitions: dict | None = None,
    sample: int | None = None,
    max_concurrency: int = DEFAULT_SUMMARIZATION_CONCURRENCY,
):
    # partitions/top level clusters
    if partitions is not None:
        num_partitions = len(partitions.keys())
        num_top_level_clusters = hierarchy[-1]
        if num_partitions != num_top_level_clusters:
            warnings.warn(
                f"Number of partitions ({num_partitions}) does not match number of top-level clusters ({num_top_level_clusters})"
            )

    # load data
    logger.info(f"Loading and summarizing examples from '{dataset_name}' dataset")
    print(f"Loading dataset '{dataset_name}'...")

    examples = list(
        client.list_examples(
            dataset_name=dataset_name, limit=sample if sample else None
        )
    )
    total_examples = len(examples)
    validate_hierarchy(hierarchy, total_examples)  # Gives you an option to quit

    logger.info(f"Loaded {total_examples} total examples, generating summaries...")
    print(f"Loaded {total_examples} examples, generating summaries...")

    summaries = await summarize_all(
        examples, partitions, summary_prompt, max_concurrency=max_concurrency
    )
    summaries_by_partition = defaultdict(list)
    # Prepare to process examples by partition
    for summary in summaries:
        if summary:
            summaries_by_partition[summary["partition"]].append(summary)

    logger.info(
        f"The dataset contains the following partitions: {list(summaries_by_partition)}"
    )
    print(f"Partitions: {list(summaries_by_partition.keys())}")

    # Process partitions one at a time and append to an updates list
    all_updates = []
    combined_hierarchy = {"partitions": {}}

    for partition, cat_summaries in summaries_by_partition.items():
        example_ids = [s["example_id"] for s in cat_summaries]
        partition_examples = [e for e in examples if e.id in example_ids]
        logger.info(f"Clustering examples that belong to partition '{partition}'")
        print(f"\nProcessing partition '{partition}'...")

        try:
            partition_updates, partition_hierarchy = cluster_partition_examples(
                partition,
                partition_examples,
                cat_summaries,
                total_examples,
                hierarchy,
            )
        except Exception as e:
            logger.error(f"ERROR processing partition {partition}: {e}")
            print(f"ERROR processing partition {partition}: {e}")
            continue
        else:
            all_updates.extend(partition_updates)
            combined_hierarchy["partitions"][partition] = partition_hierarchy

            logger.info("Searching for more partitions to cluster...")
            time.sleep(1.0)

    logger.info("All partitions have been processed, clustering complete!")
    print("\nClustering complete!")
    # Save results to csvs
    save_results(all_updates, combined_hierarchy, save_path)
