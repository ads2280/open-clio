import asyncio
import time
import uuid
from logging import getLogger

import numpy as np
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client, traceable
from pydantic import BaseModel, Field
from sklearn.cluster import KMeans

from open_clio.internal import schemas
from open_clio.prompts import (
    CRITERIA,
    DEDUPLICATE_CLUSTERS_INSTR,
    NAME_CLUSTER,
    PROPOSE_CLUSTERS_INSTR,
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


def _perform_base_clustering(
    summaries, partition_k, partition, hierarchy, partition_description=""
) -> list[schemas.ClusterInfo]:
    # Safety check: must create at least 1 cluster
    if partition_k < 1:
        partition_k = 1

    embeddings = np.array(embedder.embed_documents([s["summary"] for s in summaries]))
    kmeans = KMeans(n_clusters=partition_k, random_state=42, n_init=10, max_iter=300)
    clusters = kmeans.fit_predict(embeddings)

    return _generate_cluster_descriptions(
        clusters, summaries, embeddings, partition, hierarchy, partition_description
    )


async def _retry_with_backoff(func, max_retries=3, delay=1.0):
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except Exception as e:
            last_exception = e

            if attempt < max_retries:
                current_delay = min((delay * (2**attempt)), 30)
                logger.warning(
                    f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay} seconds..."
                )
                await asyncio.sleep(current_delay)
            else:
                logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
                raise last_exception

@traceable
async def _summarize_run(
    partitions: dict[str, str],
    run: dict,
    summary_prompt: str | None = None,
) -> schemas.SummaryAndPartition:
    """Use an LLM to generate a summary for a single run and place it in the appropriate partition."""

    class ResponseFormatter(BaseModel):
        summary: str = Field(
            description="A structured summary of run that captures the main task, request, or purpose. Be specific about the subject matter or domain and include context about the purpose or use case when relevant. Do NOT include phrases like 'User requested' or 'I understand' - start directly with the action/task."
        )
        partition: str = Field(
            description=f"The main partition this run belongs to. Must be one of: {list(partitions.keys()) if partitions else ['Default']}"
        )

    # If no partitions provided, process all runs in same partition
    if not partitions:
        partitions_str = (
            "- Default: All items in the project belong to this partition by default"
        )
    # If partitions provided, append to summary prompt
    else:
        partitions_str = "\n".join(f"- {k}: {v}" for k, v in partitions.items())

    summary_prompt_w_partitions = f"{summary_prompt}\n\nProvide your summary and also select the most appropriate partition for this conversation from the provided list:\n{partitions_str}"

    structured_llm = llm.with_structured_output(ResponseFormatter, include_raw=True)

    prompt = ChatPromptTemplate(
        [
            {"role": "system", "content": "Summarize this:"},
            {"role": "user", "content": summary_prompt_w_partitions},
        ],
        template_format="mustache",
    )

    def truncate_prompt(prompt_val, max_chars=300000) -> list:
        messages = prompt_val.to_messages()
        if len(messages[1].content) > max_chars:
            content = messages[1].content
            truncated_content = (
                content[: max_chars // 2] + "\n...\n" + content[-max_chars // 2 :]
            )
            messages[1].content = truncated_content
        return messages

    chain = prompt | truncate_prompt | structured_llm

    async def _run_chain():
        result = await chain.ainvoke({"run": run})
        if result["parsed"]:
            if isinstance(run, dict) and "id" in run:
                run_id = run["id"]
            elif (
                isinstance(run, dict)
                and "metadata" in run
                and "run_id" in run["metadata"]
            ):
                run_id = run["metadata"]["run_id"]
            else:
                raise ValueError(f"Run {run} has no ID")
        else:
            raise ValueError(f"Failed to parse run {run}")

        return {
            "summary": result["parsed"].summary,
            "partition": result["parsed"].partition,
            "run_id": run_id,
        }

    try:
        return await _retry_with_backoff(_run_chain, max_retries=3, delay=1.0)
    except Exception as e:
        logger.error(f"Failed to summarize run after 3 retries: {e}")
        return {
            "summary": None,
            "partition": "Failed",
            "run_id": run["id"] if isinstance(run, dict) and "id" in run else run["metadata"]["run_id"],
        }


def _generate_cluster_descriptions(
    clusters, summaries, embeddings, partition, hierarchy, partition_description=""
) -> list[schemas.ClusterInfo]:
    cluster_info = []
    num_clusters = max(clusters) + 1

    for idx in range(num_clusters):
        cluster_mask = clusters == idx
        cluster_summary_infos = [
            summary for summary, mask in zip(summaries, cluster_mask) if mask
        ]
        cluster_summaries = [s["summary"] for s in cluster_summary_infos]

        contrastive_summaries = _get_contrastive_summaries(
            cluster_mask, embeddings, summaries
        )

        cluster_id = uuid.uuid4()
        name, description = _generate_single_cluster_description(
            cluster_summaries,
            contrastive_summaries,
            cluster_id,
            partition,
            hierarchy,
            partition_description,
        )
        cluster_info.append(
            {
                "name": name,
                "description": description,
                "size": len(cluster_summaries),
                "summaries": [s["summary"] for s in cluster_summary_infos],
                "run_ids": [s["run_id"] for s in cluster_summary_infos],
                "partition": partition,
                "id": cluster_id,
            }
        )
        logger.info(
            f"Level 0 Cluster {cluster_id}: {description} ({len(cluster_summaries)} items)"
        )
    return cluster_info


def _get_contrastive_summaries(cluster_mask, embeddings, summaries):
    """
    Use up to 50 runs nearest to but outside of this cluster to explain what differentiates it from other clusters.
    """
    # Get contrastive ex (still within this partition)
    cluster_embeddings = embeddings[cluster_mask]
    cluster_centroid = np.mean(cluster_embeddings, axis=0)

    # Get distances from centroid to all non-cluster points within partition
    non_cluster_mask = ~cluster_mask
    non_cluster_embeddings = embeddings[non_cluster_mask]
    non_cluster_summaries = [
        summaries[i] for i in range(len(summaries)) if not cluster_mask[i]
    ]

    if len(non_cluster_summaries) > 0:
        # Calculate distances to centroid
        distances = np.linalg.norm(non_cluster_embeddings - cluster_centroid, axis=1)

        # Get closest non-cluster summaries
        n_contrastive = min(50, len(non_cluster_summaries))
        nearest_indices = np.argsort(distances)[:n_contrastive]
        contrastive_summaries = [
            non_cluster_summaries[i]["summary"] for i in nearest_indices
        ]
    else:
        contrastive_summaries = []

    return contrastive_summaries


def _generate_single_cluster_description(
    cluster_summaries,
    contrastive_summaries,
    cluster_id,
    partition,
    hierarchy,
    partition_description="",
):
    # If this is a single-level hierarchy with a non-Default partition,
    # skip LLM call and use partition name/description directly
    if len(hierarchy) == 1 and partition != "Default":
        name = partition
        summary = (
            partition_description
            if partition_description
            else f"All items in {partition} partition"
        )
        return name, summary

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

    cluster_sample = "\n".join(cluster_sample)
    contrastive_sample = "\n".join(contrastive_sample)

    prompt = NAME_CLUSTER.format(
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
        logger.error(f"Error: {e}")
        name = f"Cluster {cluster_id}"
        summary = "Error generating description"

    return name, summary


def _embed_cluster_descriptions(current_clusters):
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


def _generate_neighborhoods(cluster_embeddings, num_clusters, target_clusters):
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


def _propose_clusters_from_neighborhoods(
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
            cluster_ids[i]
            for i in range(len(cluster_ids))
            if bool(neighborhood_mask[i])
        ]
        neighborhood_clusters = {
            cid: current_clusters[cid] for cid in neighborhood_cluster_ids
        }

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
        proposing_assistant_prompt = """I understand. I'll evaluate the clusters and provide higher-level cluster names that could encompass multiple sub-clusters. 
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
            if "<answer>" in content and "</answer>" in content:
                ans_start = content.find("<answer>")
                ans_end = content.find("</answer>")
                ans_text = content[ans_start:ans_end].strip()

                for line in ans_text.split("\n"):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith("-")):
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
            time.sleep(1.0)
    logger.info(f"Proposed clusters: {proposed}")
    return proposed


def _deduplicate_proposed_clusters(proposed, target_clusters):
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
