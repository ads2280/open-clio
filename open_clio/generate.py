# updated for new ds
from langsmith import Client
import anthropic
from langsmith import wrappers
import numpy as np
import pandas as pd
import json
import argparse
from langchain_openai import OpenAIEmbeddings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time
import random
import os
from prompts import (
    CRITERIA,
    NAME_CLUSTER_INSTR,
    PROPOSE_CLUSTERS_INSTR,
    DEDUPLICATE_CLUSTERS_INSTR,
    ASSIGN_CLUSTER_INSTR,
    RENAME_CLUSTER_INSTR,
    SUMMARIZE_INSTR,
)

from collections import defaultdict
from typing import Optional
import asyncio
from anthropic import AsyncAnthropic
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field


class ResponseFormatter(BaseModel):
    summary: str = Field(
        description="A structured summary of the support conversation in the format: 'User requested [ISSUE_TYPE] help with [SPECIFIC_PRODUCT] for [LANGUAGE/FRAMEWORK] implementation'"
    )
    category: str = Field(
        description="The main product category this support request belongs to. Must be one of: 'LangChain OSS', 'LangSmith product', 'LangGraph OSS', 'LangGraph Platform/Studio', 'LangSmith deployment', 'Admin/Account management', 'Unrelated'"
    )


client = Client()
claude = wrappers.wrap_anthropic(anthropic.Anthropic())
embedder = OpenAIEmbeddings(model="text-embedding-3-small")  # openai embeddings
async_claude = AsyncAnthropic()
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514", temperature=0.2, max_tokens=100
).with_structured_output(ResponseFormatter)


async def generate_single_summary(
    example, semaphore, counter, total_examples, partitions
):
    """Use an LLM to generate a summary for a single example."""
    async with semaphore:
        conversation_text = str(example.inputs)

        # If no partitions provided, all in same category
        if not partitions:
            partitions_str = (
                "- Default: All items in the dataset belong to this category by default"
            )
        else:
            partitions_str = "\n".join(f"- {k}: {v}" for k, v in partitions.items())

        summarize_instr = SUMMARIZE_INSTR.format(partitions=partitions_str)

        messages = [
            {
                "role": "user",
                "content": f"The following is a conversation between an AI assistant and a user:\n\n{conversation_text}",
            },
            {"role": "assistant", "content": "I understand."},
            {
                "role": "user",
                "content": f"{summarize_instr}",
            },
            {
                "role": "assistant",
                "content": "Sure, I'll analyze this conversation and provide a structured summary: <answer>",
            },
        ]
        current_count = counter[0]
        counter[0] += 1
        print(f"processing example {current_count}/{total_examples} (ID: {example.id})")
        start_time = time.time()

        try:
            response = await llm.ainvoke(messages)

            # With structured output, response has summary and category fields
            res = response.summary
            category = response.category

        except Exception as e:
            print(f"Error processing example {example.id}: {e}")
            res = "Error extracting summary"
            category = "Unknown"

        print(
            f"Processed example {current_count}/{total_examples} (ID: {example.id}) in {time.time() - start_time:.2f}s"
        )
        print(f"Summary: {res}\n")

        return {
            "metadata": example.metadata,
            "inputs": example.inputs,
            "id": example.id,
            "outputs": {"summary": res, "category": category},
        }


async def summarize_all(
    examples: list, dataset_name: str, partitions: dict, max_concurrent: int = 5
):
    """Generate summaries for all examples in the dataset."""
    print(f"Generating summaries for dataset: {dataset_name}")

    total_ex = len(examples)

    semaphore = asyncio.Semaphore(max_concurrent)
    counter = [0]

    start_time = time.time()
    tasks = [
        generate_single_summary(example, semaphore, counter, total_ex, partitions)
        for example in examples
    ]
    updates = await asyncio.gather(*tasks)  # each coroutine as separate task in list

    print(f"\nSummaries generated, updating '{dataset_name}' dataset...")

    response = client.update_examples(dataset_name=dataset_name, updates=updates)

    # timing
    total_time = time.time() - start_time
    rate = total_ex / total_time if total_time > 0 else 0
    print(
        "\nSuccess, summaries complete! Reloading the dataset with summaries (and categories if applicable)..."
    )
    print(f"(total time {total_time:.2f}s, {rate:.2f} iterations/second on average)")

    return updates


def perform_base_clustering(summaries, category_k, id_offset, category):
    """
    Perform the initial base clustering for a category.

    Args:
        summaries: List of summaries to cluster
        category_k: Number of clusters to create
        id_offset: Offset for cluster IDs
        category: Category name for logging

    Returns:
        tuple: (cluster_info, cluster_labels)
    """
    # generate embeddings
    embeddings = np.array(embedder.embed_documents(summaries))
    # this assumes you'll never have less categories than k, can add a min(k, len(summaries)) check l8er

    # apply kmeans clustering
    kmeans = KMeans(n_clusters=category_k, random_state=42, n_init=10, max_iter=300)
    cluster_labels_raw = kmeans.fit_predict(embeddings)
    cluster_labels = cluster_labels_raw + id_offset
    if len(np.unique(cluster_labels_raw)) >= 2:
        silhouette = silhouette_score(embeddings, cluster_labels_raw)
        # print(f"silhouette score: {silhouette}")
    else:
        # print("Skipping silhouette score (only 1 cluster)")
        pass
    print(f"unique cluster labels: {np.unique(cluster_labels)}")
    print(f"silhoutte score: {silhouette}")

    # generate descriptions for all clusters
    cluster_info = generate_cluster_descriptions(
        cluster_labels, summaries, embeddings, category
    )

    return cluster_info, cluster_labels


def deduplicate_base_clusters(cluster_info, category_ktop):
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

    print(f"Embedding {len(cluster_ids)} cluster descriptions...")
    cluster_embeddings = np.array(embedder.embed_documents(cluster_names_descriptions))
    time.sleep(1.0)  # Sleep after embedding step

    return cluster_embeddings, cluster_ids


def generate_neighborhoods(cluster_embeddings, num_clusters):
    """
    Step 2: Generate neighborhoods using k-means clustering.

    Args:
        cluster_embeddings: Embeddings of cluster descriptions
        num_clusters: Number of current clusters

    Returns:
        tuple: (neighborhood_labels, k_neighborhoods)
    """
    k_neighborhoods = min(6, num_clusters // 2)  # Ensure reasonable neighborhoods
    k_neighborhoods = max(2, k_neighborhoods)  # At least 2 neighborhoods

    print(f"Proposing {k_neighborhoods} higher level clusters...")

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
            min_cpn=int(0.5 * clusters_per_neighborhood),
            max_cpn=int(1.5 * clusters_per_neighborhood),
        )
        proposing_assistant_prompt = """ I understand. I'll evaluate the clusters and provide higher-level
cluster names that could encompass multiple sub-clusters within the LangChain ecosystem. 
<scratchpad>"""

        try:
            response = claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=1.0,  # in clio
                messages=[
                    {"role": "user", "content": proposing_user_prompt},
                    {"role": "assistant", "content": proposing_assistant_prompt},
                ],
            )

            proposed_names = []
            content = response.content[0].text
            # extract answer section
            if "<answer>" in content and "</answer>" in content:
                ans_start = content.find("<answer>")
                ans_end = content.find("</answer>")
                ans_text = content[ans_start:ans_end].strip()

                # we should have a numbered list to parse, for ex:
                # 1. [First higher-level cluster name]
                # 2. [Second higher-level cluster name]
                # 3. [Third higher-level cluster name]
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
                        proposed_names.append(name)  # we now have proposed_names
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
            # print(f"Neighborhood {neighborhood_id} proposed: {proposed_names}")
            time.sleep(1.0)  # Sleep after each neighborhood proposal

        except Exception as e:
            print(f"Error proposing clusters for neighborhood {neighborhood_id}: {e}")
            time.sleep(1.0)  # Sleep even on error to respect rate limits

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
    print(f"Deduplicating {len(proposed)} proposed clusters")
    if len(proposed) == 0:
        print("ERROR: No clusters were proposed!")
        return []
    if len(proposed) <= target_clusters:
        return proposed

    cluster_text = "\n".join([f"<cluster>{name}</cluster>" for name in proposed])
    deduplicating_user_prompt = DEDUPLICATE_CLUSTERS_INSTR.format(
        cluster_text=cluster_text,
        target_clusters=target_clusters,
        clusters_per_neighborhood=target_clusters // 2,  # Approximate
        criteria=CRITERIA,
        min_cpn=int(0.5 * target_clusters),
        max_cpn=int(1.5 * target_clusters),
    )

    deduplicating_assistant_prompt = f"""
I understand. I'll deduplicate the cluster names into approximately {target_clusters} names.
<scratchpad>"""

    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            temperature=1.0,
            messages=[
                {"role": "user", "content": deduplicating_user_prompt},
                {
                    "role": "assistant",
                    "content": deduplicating_assistant_prompt,
                },
            ],
        )
        content = response.content[0].text
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
            print("Warning: Could not parse deduplicated clusters")
            deduplicated = proposed[:target_clusters]

    except Exception as e:
        print(f"Error deduplicating clusters: {e}")
        deduplicated = proposed[:target_clusters]

    print(f"Final deduplicated clusters: {deduplicated}")
    time.sleep(2.0)  # Sleep after deduplication step
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
    print(
        f"Assigning {len(current_clusters)} clusters to {len(deduplicated)} higher-level clusters..."
    )
    assignments = {}
    # Clio randomly shuffles higher level cluster names to avoid order bias (what?)
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
            response = claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=1.0,
                messages=[
                    {"role": "user", "content": assign_user_prompt},
                    {"role": "assistant", "content": assign_assistant_prompt},
                    {"role": "user", "content": assign_user_prompt_2},
                    {"role": "assistant", "content": assign_assistant_prompt_2},
                ],
            )
            content = response.content[0].text
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
            print(f"Error assigning cluster {cluster_id}: {e}")
            assignments[cluster_id] = deduplicated[0]

        time.sleep(1.0)  # Sleep after each cluster assignment

    return assignments


def rename_higher_level_clusters(current_clusters, assignments, level, category):
    """
    Step 6: Rename higher level clusters based on assignments.

    Args:
        current_clusters: Dictionary of current cluster information
        assignments: Mapping of cluster_id to assigned higher-level cluster name
        deduplicated: List of deduplicated higher-level cluster names
        level: Current hierarchy level
        category: Category name

    Returns:
        dict: New level clusters with names and descriptions
    """
    print("Renaming higher-level clusters based on assignments...")
    new_lvl_clusters = {}

    # group clusters by their assigned HL cluster
    cluster_groups = {}
    for cluster_id, assigned_hl in assignments.items():
        if assigned_hl not in cluster_groups:
            cluster_groups[assigned_hl] = []
        cluster_groups[assigned_hl].append(cluster_id)

    for hl_id, (hl_name, member_cluster_ids) in enumerate(cluster_groups.items()):
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
            response = claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=1.0,
                messages=[
                    {"role": "user", "content": renamingHL_user_prompt},
                    {"role": "assistant", "content": renamingHL_assistant_prompt},
                ],
            )
            content = response.content[0].text

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
            print(f"Error renaming cluster {hl_name}: {e}")
            name = f"Level {level} Cluster {hl_id}"
            summary = "Summary generation failed"

        new_lvl_clusters[hl_id] = {
            "name": name,
            "description": summary,
            "member_clusters": member_cluster_ids,
            "total_size": total_size,
            "size": len(member_cluster_ids),
            "category": category,
        }

        print(
            f"Level {level} Cluster {hl_id}: {name} ({len(member_cluster_ids)} sub-clusters, {total_size} total items)"
        )
        time.sleep(1.0)  # Sleep after each higher-level cluster renaming

    return new_lvl_clusters


def generate_cluster_descriptions(cluster_labels, summaries, embeddings, category):
    cluster_info = {}
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_summaries = [
            summaries[i] for i in range(len(summaries)) if cluster_mask[i]
        ]

        # get contrastive examples
        contrastive_summaries = get_contrastive_summaries(
            cluster_mask, embeddings, summaries
        )

        # use them to generate the description for this cluster
        name, summary = generate_single_cluster_description(
            cluster_summaries, contrastive_summaries, cluster_id
        )

        cluster_info[cluster_id] = {
            "name": name,
            "description": summary,
            "size": len(cluster_summaries),
            "summaries": cluster_summaries,
            "category": category,
        }

        print(f"Cluster {cluster_id}: {name} ({len(cluster_summaries)} items)")
        time.sleep(1.0)  # Increased sleep time for base cluster generation

    return cluster_info


def get_contrastive_summaries(cluster_mask, embeddings, summaries):
    """
    Use up to 50 examples nearest to but outside of this cluster to explain what differentiates it from other clusters.
    """
    # get contrastive ex (still within this category)
    cluster_embeddings = embeddings[cluster_mask]
    cluster_centroid = np.mean(cluster_embeddings, axis=0)

    # get distances from centroid to all non-cluster points within category
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
        contrastive_summaries = [non_cluster_summaries[i] for i in nearest_indices]
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
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=1.0,
            messages=[
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": """Sure, I will provide a clear, precise, and accurate summary and name for
    this cluster. I will be descriptive and assume neither good nor bad faith. Here
    is the summary, which I will follow with the name: <summary>""",
                },
            ],
        )
        content = response.content[0].text
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


def cluster_category_examples(
    category, category_examples, total_examples, id_offset, hierarchy
):
    """
    Orchestrates hierarchical clustering for each user-defined category/partition.
    Defaults to processing all examples if no categories/partitions provided.
    """

    # Extract summaries and example IDs for this category
    summaries = []
    example_ids = []
    for example in category_examples:
        summaries.append(example.outputs["summary"])
        example_ids.append(example.id)

    # proportional cluster sizes
    category_k = int(hierarchy[0] * len(category_examples) / total_examples)
    category_k = min(category_k, len(summaries))
    category_k = max(1, category_k)
    category_ktop = int(hierarchy[-1] * len(category_examples) / total_examples)

    print(
        f"Examples: {len(category_examples)}\nBase clusters: {category_k}\nTarget top clusters: {category_ktop}"
    )

    # Perform base clustering
    cluster_info, cluster_labels = perform_base_clustering(
        summaries, category_k, id_offset, category
    )
    # deduplicate_base_clusters(cluster_info, category_ktop) at some point

    print("\nBuilding the next level of clusters...")

    # start hierarchical clustering
    category_hierarchy = {"level_0": cluster_info, "max_level": 0}
    current_clusters = cluster_info
    n_base = len(current_clusters)

    # Track example assignments at each level
    example_assignments = {
        "level_0": {
            example_id: cluster_id
            for example_id, cluster_id in zip(example_ids, cluster_labels)
        }
    }

    # Use user-provided hierarchy instead of geometric progression
    # hierarchy = [k, x, ktop] where k is base clusters, x is intermediate level, ktop is target
    levels = len(hierarchy)
    level_sizes = hierarchy  # TODO - decide if we skip the first element (base k) since we already have n_base clusters

    # Scale the hierarchy proportionally to this category's size
    category_ratio = len(category_examples) / total_examples
    scaled_level_sizes = [max(2, int(size * category_ratio)) for size in level_sizes]

    full_hierarchy = [n_base] + scaled_level_sizes
    print(f"(target hierarchy for this dataset: {full_hierarchy})")

    # Build clusters for each level in the hierarchy
    for level in range(1, levels):
        if len(current_clusters) <= scaled_level_sizes[level - 1]:
            print(f"stopping at level {level - 1} bc only {len(current_clusters)} left")
            break

        # print(f"creating level {level}, targeting {scaled_level_sizes[level - 1]} clusters")

        # 1) embed clusters
        cluster_embeddings, cluster_ids = embed_cluster_descriptions(current_clusters)

        # 2) generate neighbourhoods using k-means
        neighborhood_labels, k_neighborhoods = generate_neighborhoods(
            cluster_embeddings, len(current_clusters)
        )

        # 3) propose new clusters for each neighborhood
        target_clusters = scaled_level_sizes[level - 1]
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
        assignments = assign_clusters_to_higher_level(current_clusters, deduplicated)

        # 6) rename higher level clusters based on assignments
        new_lvl_clusters = rename_higher_level_clusters(
            current_clusters, assignments, level, category
        )

        # Track example assignments for this level
        example_assignments[f"level_{level}"] = {}
        for example_id, base_cluster_id in example_assignments["level_0"].items():
            # Find which higher-level cluster this base cluster was assigned to
            if base_cluster_id in assignments:
                higher_level_cluster_name = assignments[base_cluster_id]
                # Find the cluster ID for this higher-level cluster name
                for hl_cluster_id, hl_cluster_info in new_lvl_clusters.items():
                    if hl_cluster_info["name"] == higher_level_cluster_name:
                        example_assignments[f"level_{level}"][example_id] = (
                            hl_cluster_id
                        )
                        break

        category_hierarchy[f"level_{level}"] = new_lvl_clusters
        category_hierarchy["max_level"] = level
        current_clusters = new_lvl_clusters

        print(f"Level {level} complete, checking if more levels are needed...")
        # after this loop term, we have new_lvl_clusters dict with new level clusters

    print(
        f"No more levels needed, hierarchical clustering complete for category '{category}'!"
    )

    category_updates = []
    for i, example_id in enumerate(example_ids):
        example = category_examples[i]

        # Build nested clustering structure
        clustering = {}
        for level_key, level_assignments in example_assignments.items():
            if example_id in level_assignments:
                cluster_id = level_assignments[example_id]
                # Find cluster info for this level
                if level_key == "level_0":
                    cluster_info_for_level = cluster_info
                else:
                    cluster_info_for_level = category_hierarchy[level_key]

                if cluster_id in cluster_info_for_level:
                    clustering[level_key] = {
                        "id": int(cluster_id),
                        "name": cluster_info_for_level[cluster_id]["name"],
                    }

        update = {
            "id": example_id,
            "metadata": example.metadata,
            "inputs": example.inputs,
            "outputs": {
                "summary": example.outputs["summary"],
                "category": example.outputs["category"],
                "clustering": clustering,
            },
        }
        category_updates.append(update)

    # calculate next cluster id offset
    max_cluster_id = max(cluster_info.keys()) if cluster_info else id_offset
    next_offset = max_cluster_id + 1

    return category_updates, category_hierarchy, next_offset


def save_results(client, dataset_name, all_updates, combined_hierarchy, save_path):
    # print results summary
    print("\nOverview of clustering results:")
    for category, hierarchy in combined_hierarchy["categories"].items():
        print(f"\nCategory: {category}")
        print(f"Base clusters: {len(hierarchy['level_0'])}")
        if hierarchy["max_level"] > 0:
            for level in range(1, hierarchy["max_level"] + 1):
                print(f"Level {level} clusters: {len(hierarchy[f'level_{level}'])}")

    # update dataset with all cluster assignments
    print("\nUpdating the dataset with clustering results...")
    client.update_examples(dataset_name=dataset_name, updates=all_updates)

    os.makedirs(save_path, exist_ok=True)

    # 1. combined.csv: save combined examples with full hierarchical clustering info
    print(f"\nSaving results to {save_path}...")
    examples_data = []
    for update in all_updates:
        clustering = update["outputs"]["clustering"]

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
        top_level = max(clustering.keys()) if clustering else "level_0"
        top_cluster_id = clustering.get(top_level, {}).get("id", None)
        top_cluster_name = clustering.get(top_level, {}).get("name", "")

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
            "category": update["outputs"]["category"],
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

        examples_data.append(row_data)  # examples_data has row_data for every example

    examples_df = pd.DataFrame(examples_data)
    examples_df.to_csv(f"{save_path}/combined.csv", index=False)
    # looks like: example_id,full_example,summary,category,base_cluster_id,base_cluster_name, [intermediates], top_cluster_id,top_cluster_name
    print(f"Saved {len(examples_data)} examples to combined.csv")

    # 2. csv by level: Save ALL clusters from ALL levels across ALL categories
    all_clusters = {"level_0": [], "level_1": [], "level_2": []}

    for category, cat_hierarchy in combined_hierarchy["categories"].items():
        for level in range(cat_hierarchy["max_level"] + 1):
            level_key = f"level_{level}"
            if level_key in cat_hierarchy:
                for cluster_id, cluster_data in cat_hierarchy[level_key].items():
                    row = {
                        "cluster_id": cluster_id,
                        "name": cluster_data.get("name", ""),
                        "description": cluster_data.get("description", ""),
                        "size": cluster_data.get("size", 0),
                        "category": category,
                    }

                    # Add level-specific fields
                    if "total_size" in cluster_data:
                        row["total_size"] = cluster_data["total_size"]
                    if "member_clusters" in cluster_data:
                        row["member_clusters"] = str(cluster_data["member_clusters"])

                    all_clusters[level_key].append(row)

    # Save each level
    for level_name, cluster_list in all_clusters.items():
        if not cluster_list:
            continue

        df = pd.DataFrame(cluster_list)
        df = df.sort_values("size", ascending=False)
        df.to_csv(f"{save_path}/{level_name}_clusters.csv", index=False)
        print(f"Saved {len(cluster_list)} clusters to {level_name}_clusters.csv")

    print(f"\nSee {save_path} for csv files.")
    print("\nthe end")


def load_config(config_path="config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(
            f"Config file {config_path} not found. Please create {config_path} with your configuration."
        )
        exit(1)


def validate_hierarchy(hierarchy, n_examples):
    """check if hierarchy makes sense"""
    # number of base clusters
    suggested_max_k = min(int(np.sqrt(n_examples)), n_examples // 3)
    if hierarchy[0] > suggested_max_k:
        suggested_k = suggested_max_k
        print(
            f"Warning: {hierarchy[0]} base clusters seems like too many for {n_examples} examples"
        )
        print(f"Consider starting with <={suggested_k} clusters")

    # decreasing numbers
    for i in range(len(hierarchy) - 1):
        if hierarchy[i] <= hierarchy[i + 1]:
            print(
                f"Warning: Level {i} has {hierarchy[i]} clusters, level {i + 1} has {hierarchy[i + 1]}"
            )


def generate_clusters(
    dataset_name: str,
    save_path: str,  # combined.csv
    hierarchy: list,
    partitions: Optional[dict] = None,
    sample: Optional[int] = None,
):
    # load data
    print(f"Loading and summarizing examples from '{dataset_name}' dataset")
    validate_hierarchy(hierarchy, total_examples)  # Gives you an option to quit

    examples = list(
        client.list_examples(
            dataset_name=dataset_name, limit=sample if sample else None
        )
    )
    total_examples = len(examples)
    print(f"Loaded {total_examples} total examples, generating summaries...")

    asyncio.run(summarize_all(examples, dataset_name, partitions, max_concurrent=5))
    time.sleep(30)  # TODO fix

    # Reload data to get the updated examples with summaries
    examples = list(
        client.list_examples(
            dataset_name=dataset_name, limit=sample if sample else None
        )
    )

    # Prepare to process examples by category
    examples_by_category = defaultdict(list)
    for example in examples:
        examples_by_category[example.outputs["category"]].append(example)

    print(
        f"\nThe dataset contains the following categories: {list(examples_by_category.keys())}"
    )

    # Process categories one at a time and append to an updates list
    all_updates = []
    combined_hierarchy = {"categories": {}}
    id_offset = 0

    skip_categories = [
        "Error extracting category",
        "<UNKNOWN>",
        "Unknown",
    ]  # one of these should always be what we provide as a 'misc' category TODO

    for (
        category,
        category_examples,
    ) in examples_by_category.items():  # enumerate, can count later >> TODO
        if category in skip_categories:
            print(
                f"\n\nSkipping problematic category: {category} ({len(category_examples)} examples)"
            )
            continue

        print(f"\n\nStarting to cluster examples that belong to category '{category}'")
        validate_hierarchy(hierarchy, len(category_examples))

        try:
            category_updates, category_hierarchy, next_offset = (
                cluster_category_examples(  # TODO - ideally its the same for anything, same set of reusable methods whether categories, no categories, whatever levle youo're add
                    category, category_examples, total_examples, id_offset, hierarchy
                )
            )
            # TODO play with updates logic --> combined.csv should be the only result

            all_updates.extend(category_updates)
            combined_hierarchy["categories"][category] = category_hierarchy  # TODO
            id_offset = next_offset

            print("Searching for more categories to cluster...")
            time.sleep(3.0)  # Sleep between categories

        except Exception as e:
            print(
                f"ERROR processing category {category}: {e}"
            )  # TODO better error handling
            continue

    print("\nAll categories have beenprocessed, clustering complete!")

    # Save results to csvs
    save_results(client, dataset_name, all_updates, combined_hierarchy, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Clio clustering")
    parser.add_argument("--dataset", help="Override dataset name")
    args = parser.parse_args()

    config = load_config()

    if args.dataset:
        config["dataset_name"] = args.dataset

    print("Starting Clio clustering...")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Save path: {config['save_path']}")
    print(f"Hierarchy (number of examples at each level): {config['hierarchy']}\n")

    generate_clusters(
        dataset_name=config["dataset_name"],
        save_path=config["save_path"],
        hierarchy=config["hierarchy"],
        partitions=config["partitions"],
        sample=config["sample"],
    )


# TODO bagatur/code review note,
# neo recs,

# and make sure it works - cna prob start
