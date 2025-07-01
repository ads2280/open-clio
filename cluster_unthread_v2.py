# updated for new ds

from langsmith import Client
import anthropic
from langsmith import wrappers
import numpy as np
import pandas as pd

# from sentence_transformers import SentenceTransformer
from langchain_openai import OpenAIEmbeddings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math
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
)
from collections import defaultdict


client = Client()
claude = wrappers.wrap_anthropic(anthropic.Anthropic())
# embedder = SentenceTransformer("all-mpnet-base-v2")
embedder = OpenAIEmbeddings(model="text-embedding-3-small")
examples = list(client.list_examples(dataset_name="unthread-data"))
examples_by_category = defaultdict(list)
total_examples = 0
# summaries = []
# example_ids = []

for example in examples:
    # summaries.append(example.outputs["request"])  # i.e. summary from spreadsheet
    # example_ids.append(example.id)
    examples_by_category[example.outputs["category"]].append(example)
    total_examples += 1

k = math.sqrt(len(examples))
ktop = 8
levels = 3
avg_clusters_per_neighborhood = 10
m_nearest = 5
min_clusters_per_category = 2
min_examples_for_clustering = 5
max_summaries_for_description = 15
n_contrastive = 50


def cluster_category_examples(category, category_examples, id_offset):
    print(f"PROCESSING CATEGORY: {category}")

    # first, do base clusters

    # Extract summaries and example IDs for this category
    summaries = []
    example_ids = []
    for example in category_examples:
        summaries.append(example.outputs["request"])
        example_ids.append(example.id)

    # proportional cluster sizes
    category_k = int(k * len(category_examples) / total_examples)
    category_k = min(category_k, len(summaries))
    category_k = max(1, category_k)
    category_ktop = int(ktop * len(category_examples) / total_examples)

    print(
        f"Examples: {len(category_examples)}\nBase clusters: {category_k}\nTarget top clusters: {category_ktop}"
    )

    # embed
    embeddings = np.array(embedder.embed_documents(summaries))
    print(f"embeddings generated, shape: {embeddings.shape}")
    # this assumes you'll never have less categories than k, can add a min(k, len(summaries)) check l8er

    # kmeans
    kmeans = KMeans(n_clusters=category_k, random_state=42, n_init=10, max_iter=300)
    cluster_labels_raw = kmeans.fit_predict(embeddings)
    cluster_labels = cluster_labels_raw + id_offset
    if len(np.unique(cluster_labels_raw)) >= 2:
        silhouette = silhouette_score(embeddings, cluster_labels_raw)
        print(f"silhouette score: {silhouette}")
    else:
        print("Skipping silhouette score (only 1 cluster)")
    print(f"cluster labels: {cluster_labels}")

    # generate descriptions for all clusters
    cluster_info = {}
    unique_clusters = np.unique(cluster_labels)

    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_summaries = [
            summaries[i] for i in range(len(summaries)) if cluster_mask[i]
        ]

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
            distances = np.linalg.norm(
                non_cluster_embeddings - cluster_centroid, axis=1
            )

            # get closest non-cluster summaries
            n_contrastive = min(50, len(non_cluster_summaries))
            nearest_indices = np.argsort(distances)[:n_contrastive]
            contrastive_summaries = [non_cluster_summaries[i] for i in nearest_indices]
        else:
            contrastive_summaries = []

        # use to generate the description
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

        criteria = CRITERIA
        prompt = NAME_CLUSTER_INSTR.format(
            cluster_sample=cluster_sample,
            contrastive_sample=contrastive_sample,
            criteria=criteria,
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

        cluster_info[cluster_id] = {
            "name": name,
            "description": summary,
            "size": len(cluster_summaries),
            "summaries": cluster_summaries,
            "category": category,
        }

        print(f"Cluster {cluster_id}: {name} ({len(cluster_summaries)} items)")
        time.sleep(1.0)  # Increased sleep time for base cluster generation

    print(f"\nBase clustering complete for {category}!!")

    # starting hierarchical clustering for this category
    hierarchy = {"level_0": cluster_info, "max_level": 0}
    current_clusters = cluster_info
    n_base = len(current_clusters)

    # edge case - if n_base <= 1?
    # clio's geometric progression algo for no. of clusters at each level
    if levels == 2:
        level_sizes = [category_ktop]
    else:
        ratio = (category_ktop / n_base) ** (1 / (levels - 1))
        level_sizes = []
        for level in range(1, levels):
            n_level = int(n_base * (ratio**level))
            level_sizes.append(max(2, n_level))
        level_sizes.append(category_ktop)

    print(f"planned hierarchy sizes for {category}: {n_base} + {level_sizes}")

    # build hierarchy lvl by lvl
    for level in range(1, levels):
        if len(current_clusters) <= level_sizes[level - 1]:
            print(f"stopping at level {level - 1} bc only {len(current_clusters)} left")
            break

        print(f"creating level {level}, targeting {level_sizes[level - 1]} clusters")

        # 1) embed clusters
        # errm 1) to 6) should rly be separate methods
        cluster_names_descriptions = []
        cluster_ids = []

        for cluster_id, info in current_clusters.items():
            text = f"{info['name']}: {info['description']}"
            cluster_names_descriptions.append(text)
            cluster_ids.append(cluster_id)

        print(f"embedding {len(cluster_ids)} cluster descriptions...")
        cluster_embeddings = embedder.encode(
            cluster_names_descriptions, show_progress_bar=True
        )
        time.sleep(1.0)  # Sleep after embedding step

        # 2) generate neighbourhoods using k-means
        k_neighborhoods = min(
            6, len(current_clusters) // 2
        )  # Ensure reasonable neighborhoods
        k_neighborhoods = max(2, k_neighborhoods)  # At least 2 neighborhoods

        print(f"making {k_neighborhoods} neighborhoods")

        kmeans_nbh = KMeans(
            n_clusters=k_neighborhoods, random_state=42, n_init=10, max_iter=300
        )
        neighborhood_labels = kmeans_nbh.fit_predict(cluster_embeddings)

        # 3) # 3) propose new clusters for each neighborhood
        proposed = []
        target_clusters = level_sizes[level - 1]
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
                criteria=criteria,
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
                print(f"Neighborhood {neighborhood_id} proposed: {proposed_names}")
                time.sleep(1.0)  # Sleep after each neighborhood proposal

            except Exception as e:
                print(
                    f"Error proposing clusters for neighborhood {neighborhood_id}: {e}"
                )
                time.sleep(1.0)  # Sleep even on error to respect rate limits

        # 4) dedpulicate across neighborhoods
        print(f"Deduplicating {len(proposed)} proposed clusters")
        if len(proposed) == 0:
            print("ERROR: No clusters were proposed!")
            break
        if len(proposed) <= target_clusters:
            deduplicated = proposed
        else:
            cluster_text = "\n".join(
                [f"<cluster>{name}</cluster>" for name in proposed]
            )
            deduplicating_user_prompt = DEDUPLICATE_CLUSTERS_INSTR.format(
                cluster_text=cluster_text,
                target_clusters=target_clusters,
                clusters_per_neighborhood=clusters_per_neighborhood,
                criteria=criteria,
                min_cpn=int(0.5 * target_clusters),
                max_cpn=int(1.5 * target_clusters),
            )
            # TODO why was this failing earlier

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
                print(f"DEBUG - Proposed clusters: {proposed}")
                print(f"DEBUG - Target clusters: {target_clusters}")
                print(f"DEBUG - Will deduplicate: {len(proposed) > target_clusters}")
                deduplicated = []
                if "<answer>" in content and "</answer>" in content:
                    ans_start = content.find("<answer>") + 8
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
                            deduplicated.append(name)
                else:
                    print("Warning: Could not parse deduplicated clusters")
                    deduplicated = proposed[:target_clusters]

            except Exception as e:
                print(f"Error deduplicating clusters: {e}")
                deduplicated = proposed[:target_clusters]

        print(f"Final deduplicated clusters: {deduplicated}")
        time.sleep(2.0)  # Sleep after deduplication step

        # 5) clusters to higher level clusters
        print(
            f"Assigning {len(current_clusters)} clusters to {len(deduplicated)} higher-level clusters..."
        )
        assignments = {}
        # Clio randomly shuffles higher level cluster names to avoid order bias (what?)
        shuffled = deduplicated.copy()
        random.shuffle(shuffled)
        higher_level_text = "\n".join(
            [f"<cluster>{name}</cluster>" for name in shuffled]
        )

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

        # 6) rename higher levelclusters based on assignments
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
                cluster_list_text=cluster_list_text, criteria=criteria
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

        hierarchy[f"level_{level}"] = new_lvl_clusters
        hierarchy["max_level"] = level
        current_clusters = new_lvl_clusters

        print(f"Level {level} complete: {len(new_lvl_clusters)} clusters created")
        # after this loop term, we have new_lvl_clusters dict with new level clusters

    print(f"Hierarchical clustering complete for category {category}!")

    # NEW -- create updates for this category
    category_updates = []
    for i, (example_id, cluster_id) in enumerate(zip(example_ids, cluster_labels)):
        cluster_name = cluster_info[cluster_id]["name"]
        example = category_examples[i]

        update = {
            "id": example_id,
            "metadata": example.metadata,
            "inputs": example.inputs,
            "outputs": {
                "request": example.outputs["request"],
                "category": example.outputs["category"],
                "cluster_id": int(cluster_id),
                "cluster_name": cluster_name,
            },
        }
        category_updates.append(update)

    # calculate next cluster id offset
    max_cluster_id = max(cluster_info.keys()) if cluster_info else id_offset  # TODO
    next_offset = max_cluster_id + 1

    return category_updates, hierarchy, next_offset


print(f"Loaded {total_examples} total examples")
print(f"Categories: {list(examples_by_category.keys())}")
print(f"starting to cluster, using K={k} clusters for {len(examples)} summaries")

# main execution: process each category separately
all_updates = []
combined_hierarchy = {"categories": {}}
id_offset = 0

skip_categories = ["Error extracting category", "<UNKNOWN>"]

for category, category_examples in examples_by_category.items():
    if category in skip_categories:
        print(
            f"\n\nSkipping problematic category: {category} ({len(category_examples)} examples)"
        )
        continue

    print(f"\n\nStarting processing for category: {category}")
    print(
        f"Using K={int(k * len(category_examples) / total_examples)} clusters for {len(category_examples)} summaries"
    )

    try:
        category_updates, category_hierarchy, next_offset = cluster_category_examples(
            category, category_examples, id_offset
        )

        all_updates.extend(category_updates)
        combined_hierarchy["categories"][category] = category_hierarchy  # TODO
        id_offset = next_offset

        print(f"Completed category {category}. Next cluster ID offset: {id_offset}")
        time.sleep(3.0)  # Sleep between categories

    except Exception as e:
        print(f"ERROR processing category {category}: {e}")
        print(f"Skipping category {category} and continuing...")
        continue


# update dataset with all cluster assignments
print("UPDATING DATASET WITH ALL CLUSTER ASSIGNMENTS...")
client.update_examples(dataset_name="unthread-data", updates=all_updates)
print("Dataset updated!")

# print results summary
print("\nRESULTS SUMMARY:")
for category, hierarchy in combined_hierarchy["categories"].items():
    print(f"\nCategory: {category}")
    print(f"Base clusters: {len(hierarchy['level_0'])}")
    if hierarchy["max_level"] > 0:
        for level in range(1, hierarchy["max_level"] + 1):
            print(f"Level {level} clusters: {len(hierarchy[f'level_{level}'])}")


# prev saving results method
print("Clustering complete! Saving Results...")
output_dir = f"category_results/sonnet4-{levels}layers-{k}base-by-category"

# prev saving results method
print("Clustering complete! Saving Results...")
output_dir = f"category_results/sonnet4-{levels}layers-{k}base-by-category"

# Create combined CSV files from all category data

os.makedirs(output_dir, exist_ok=True)

# 1. Save combined examples (this part is correct)
examples_data = []
for update in all_updates:
    examples_data.append(
        {
            "example_id": update["id"],
            "summary": update["outputs"]["request"],
            "base_cluster_id": update["outputs"]["cluster_id"],
            "cluster_name": update["outputs"]["cluster_name"],
            "category": update["outputs"]["category"],
        }
    )

examples_df = pd.DataFrame(examples_data)
examples_df.to_csv(f"{output_dir}/examples.csv", index=False)
print(f"Saved {len(examples_data)} examples to examples.csv")

# 2. Save ALL clusters from ALL levels across ALL categories
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
    df.to_csv(f"{output_dir}/{level_name}_clusters.csv", index=False)
    print(f"Saved {len(cluster_list)} clusters to {level_name}_clusters.csv")

print(f"\nSee {output_dir} for csv files.")
print("\nCOMPLETE!")

# could also just be appending to the same csv for each category that's done to check it better?

# # Before calling save_results, flatten the hierarchy
# flattened_hierarchy = {"level_0": {}, "max_level": 0}
# all_cluster_ids = []
# all_summaries = []

# for category, cat_hierarchy in combined_hierarchy["categories"].items():
#     # Combine all base clusters across categories
#     flattened_hierarchy["level_0"].update(cat_hierarchy["level_0"])

#     # Update max_level if needed
#     if cat_hierarchy["max_level"] > flattened_hierarchy["max_level"]:
#         flattened_hierarchy["max_level"] = cat_hierarchy["max_level"]
#         # Copy higher level data
#         for level in range(1, cat_hierarchy["max_level"] + 1):
#             if f"level_{level}" not in flattened_hierarchy:
#                 flattened_hierarchy[f"level_{level}"] = {}
#             flattened_hierarchy[f"level_{level}"].update(cat_hierarchy[f"level_{level}"])

# # Now call save_results with the flattened version
# dataframes = save_results(
#     hierarchy=flattened_hierarchy,
#     example_ids=[update["id"] for update in all_updates],
#     cluster_labels=[update["outputs"]["cluster_id"] for update in all_updates],
#     summaries=[update["outputs"]["request"] for update in all_updates],
#     output_dir=output_dir,
# )
# print(f"\nSee {output_dir} for csv files.")
# print("\nCOMPLETE!")

# TODO - figure out global offset
# TODO - data saving
# next could do a different base k?
