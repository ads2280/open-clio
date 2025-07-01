#updated for new ds

from langsmith import Client
import anthropic
import openai
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math
import time
import random
from save_results import save_results
import os
from prompts import CRITERIA, NAME_CLUSTER_INSTR, PROPOSE_CLUSTERS_INSTR, DEDUPLICATE_CLUSTERS_INSTR, ASSIGN_CLUSTER_INSTR, RENAME_CLUSTER_INSTR

client = Client()
claude = anthropic.Anthropic()

# load summaries from ls
examples = list(client.list_examples(dataset_name="unthread-data"))
summaries = []
example_ids = []

for example in examples:
    summaries.append(example.outputs["request"])  # i.e. summary from spreadsheet
    example_ids.append(example.id)

print(f"Loaded {len(summaries)} summaries")

# generate embeddings
embedder = SentenceTransformer("all-mpnet-base-v2")
embeddings = embedder.encode(summaries, show_progress_bar=True)
print(f"embeddings generated, shape: {embeddings.shape}")

# variable declarations - at some point make configurabl
k = 150 # changed for unthread, [unknown] in clio paper
ktop = 8  # desired no. of top level clusters, 10 in clio paper
          # this should be around the number of major product areas
L = 3  # no of levels, 3 in clio changed for unthread
avg_clusters_per_neighborhood = 10  # 40 in clio paper
m_nearest = 5  # nearest clusters for contrastive examples, 10 in clio paper

print(f"starting to cluster, using K={k} clusters for {len(summaries)} summaries")

# cluster embeddings with k-means
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)

cluster_labels = kmeans.fit_predict(embeddings)
silhouette = silhouette_score(embeddings, cluster_labels)
print(f"silhouette score: {silhouette}")  # want highest poss score

print(f"cluster labels: {cluster_labels}")

# generate descriptions for all clusters
cluster_info = {}
unique_clusters = np.unique(cluster_labels)

for cluster_id in unique_clusters:
    cluster_mask = cluster_labels == cluster_id
    cluster_summaries = [summaries[i] for i in range(len(summaries)) if cluster_mask[i]]

    # Get contrastive examples
    cluster_embeddings = embeddings[cluster_mask]
    cluster_centroid = np.mean(cluster_embeddings, axis=0)

    # get distances from centroid to all non-cluster points
    non_cluster_mask = ~cluster_mask
    non_cluster_embeddings = embeddings[non_cluster_mask]
    non_cluster_summaries = [
        summaries[i] for i in range(len(summaries)) if not cluster_mask[i]
    ]

    # calculate distances to centroid
    distances = np.linalg.norm(non_cluster_embeddings - cluster_centroid, axis=1)

    # get closest non-cluster summaries
    n_contrastive = 50  # make this a param later
    nearest_indices = np.argsort(distances)[:n_contrastive]
    contrastive_summaries = [non_cluster_summaries[i] for i in nearest_indices]

    # use to generate the description
    max_summaries = 15  # arbitrary, to avoid token lims, can change later
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

    prompt = NAME_CLUSTER_INSTR.format(cluster_sample=cluster_sample, contrastive_sample=contrastive_sample, criteria=criteria)

    starter = """Sure, I will provide a clear, precise, and accurate summary and name for
this cluster. I will be descriptive and assume neither good nor bad faith. Here
is the summary, which I will follow with the name: <summary>"""

    try:
        response = claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            temperature=1.0,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": starter},
            ],
        )

        content = response.content[0].text
        # content = response.choices[0].message.content
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
        "summaries": cluster_summaries[:10],  # just first 10 for inspection
    }

    print(f"Cluster {cluster_id}: {name} ({len(cluster_summaries)} items)")
    time.sleep(0.5)

# cluster loop complete, now preparing dataset updates
print("updating dataset with cluster assignments...")
updates = []
for i, (example_id, cluster_id) in enumerate(zip(example_ids, cluster_labels)):
    cluster_name = cluster_info[cluster_id]["name"]
    example = examples[i]

    update = {
        "id": example_id,
        "metadata": example.metadata,
        "inputs": example.inputs,
        "outputs": {
            "request": example.outputs[
                "request"
            ],  # keep existing request, can change this later when generating csvs
            "cluster_id": int(cluster_id),
            "cluster_name": cluster_name,
        },
    }
    updates.append(update)

# update all
client.update_examples(dataset_name="unthread-data", updates=updates)
print("\nDataset updated!")

# print results
print("\nResults:")
for cluster_id, info in cluster_info.items():
    print(f"\nCluster {cluster_id}: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Size: {info['size']} conversations")
    print(f"Sample summaries: {info['summaries'][:3]}")

print("\nbase cluster complete!")


# starting hierarchical clustering (G7)
# load base cluster names from ls

hierarchy = {"level_0": cluster_info, "max_level": 0}
current_clusters = cluster_info
n_base = len(current_clusters)

# clio's geometric progression algo for no. of clusters at each level
if L == 2:
    level_sizes = [ktop]  # for 2 levels, just base to top
else:
    ratio = (ktop / n_base) ** (1 / (L - 1))
    level_sizes = []
    for level in range(1, L):
        n_level = int(n_base * (ratio**level))
        level_sizes.append(max(2, n_level))  # make sure atleast 2 clusters
    level_sizes.append(ktop)

print(f"planned hierarchy sizes: {n_base} + {level_sizes}")

# build hierarchy lvl by lvl
for level in range(1, L):
    if len(current_clusters) <= level_sizes[level - 1]:
        print(f"stopping at level {level - 1} bc only {len(current_clusters)} left")
        break

    print(f"creating level {level}, targetting {level_sizes[level - 1]} clusters")

    # 1) embed clusters
    cluster_names_descriptions = []
    cluster_ids = []

    for cluster_id, info in current_clusters.items():
        # combine name and description for embedding
        text = f"{info['name']}: {info['description']}"
        cluster_names_descriptions.append(text)
        cluster_ids.append(cluster_id)

    print(f"embedding {len(cluster_ids)} cluster descriptions...")
    # Generate embeddings using SentenceTransformer for cluster descriptions
    cluster_embeddings = embedder.encode(cluster_names_descriptions, show_progress_bar=True)

    # 2) generate neighbourhoods using k-means
    k_neighborhoods = 6 
    # k_neighborhoods = max(5, len(current_clusters) // avg_clusters_per_neighborhood)
    print(
        f"making {k_neighborhoods} neighborhoods with approx {avg_clusters_per_neighborhood} clusters per neighborhood"
    )

    kmeans = KMeans(
        n_clusters=k_neighborhoods, random_state=42, n_init=10, max_iter=300
    )

    neighborhood_labels = kmeans.fit_predict(cluster_embeddings)

    # 3) now propose new clusters for each neighborhood
    proposed = []
    target_clusters = level_sizes[level - 1]
    #hardcoded for now
    clusters_per_neighborhood = 5
    # clusters_per_neighborhood = max(1, target_clusters // k_neighborhoods)

    for neighborhood_id in range(k_neighborhoods):
        neighborhood_mask = neighborhood_labels == neighborhood_id
        neighborhood_cluster_ids = [
            cluster_ids[i] for i in range(len(cluster_ids)) if neighborhood_mask[i]
        ]
        neighborhood_clusters = {
            cid: current_clusters[cid] for cid in neighborhood_cluster_ids
        }

        # find nearest m clusters outside this nbh for contrastive
        neighborhood_embeddings = cluster_embeddings[neighborhood_mask]
        neighborhood_centroid = np.mean(neighborhood_embeddings, axis=0)

        outside_mask = ~neighborhood_mask
        if np.any(outside_mask):
            outside_embeddings = cluster_embeddings[outside_mask]
            distances = np.linalg.norm(
                outside_embeddings - neighborhood_centroid, axis=1
            )
            nearest_indices = np.argsort(distances)[:m_nearest]  # was manually set to 3
            outside_cluster_ids = [
                cluster_ids[i] for i in range(len(cluster_ids)) if outside_mask[i]
            ]
            nearest_outside_ids = [outside_cluster_ids[i] for i in nearest_indices]
            nearest_outside_clusters = {
                cid: current_clusters[cid] for cid in nearest_outside_ids
            }
        else:  # nothing outside this cluster - updated min clusteres to 2 so shouldn't happen
            nearest_outside_clusters = {}

        # now actually ready to propose clusters
        # nbh_proposed = propose_clusters_for_neighborhood

        # build cluster list for prompt
        cluster_list = []
        for cluster_id, info in neighborhood_clusters.items():
            cluster_list.append(
                f"<cluster>{info['name']}: {info['description']}</cluster>"
            )

        cluster_list_text = "\n".join(cluster_list)

        proposing_user_prompt = PROPOSE_CLUSTERS_INSTR.format(cluster_list_text=cluster_list_text, clusters_per_neighborhood=clusters_per_neighborhood, criteria=criteria, min_cpn=int(0.5 * clusters_per_neighborhood), max_cpn=int(1.5 * clusters_per_neighborhood))

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
            content = response.content[0].text
            #content = response.content[0].text
            print(f"Claude response for neighborhood {neighborhood_id}:")
            print(content[:200] + "..." if len(content) > 200 else content)

            proposed_names = []
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
                print(
                    f"Warning: No names extracted from Claude response for neighborhood {neighborhood_id}"
                )
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

        except Exception:
            print(
                f"Error, could not propose clusters for neighborhood {neighborhood_id}"
            )

    # 4) deduplicate across neighborhoods
    print(f"Deduplicating {len(proposed)} proposed clusters")
    if len(proposed) == 0:
        print("ERROR: No clusters were proposed!")
    if len(proposed) <= clusters_per_neighborhood:
        deduplicated = proposed  #
    else:
        cluster_text = "\n".join(
            [f"<cluster>{name}</cluster>" for name in proposed]
        )  # for prompt
        # changed {clusters_per_neighborhood} to {target_clusters}, twice
        deduplicating_user_prompt = DEDUPLICATE_CLUSTERS_INSTR.format(cluster_text=cluster_text, target_clusters=target_clusters, clusters_per_neighborhood=clusters_per_neighborhood, criteria=criteria, min_cpn=int(0.5 * clusters_per_neighborhood), max_cpn=int(1.5 * clusters_per_neighborhood))

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
                    {"role": "assistant", "content": deduplicating_assistant_prompt},
                ],
            )
            content = response.content[0].text

            deduplicated = []
            # extract answer section
            if "<answer>" in content and "</answer>" in content:
                ans_start = content.find("<answer>") + 8
                ans_end = content.find("</answer>")
                ans_text = content[ans_start:ans_end].strip()

                # numbered list like above to parse
                for line in ans_text.split("\n"):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith("-")):
                        # remove and clean up
                        name = (
                            line.split(".", 1)[1].strip()
                            if "." in line
                            else line.strip()
                        )
                        if name.startswith("[") and name.endswith(
                            "]"
                        ):  # can I assume it always does or no?
                            name = name[1:-1]
                        deduplicated.append(name)
            else:
                print("Warning: Could not parse deduplicated clusters")
                deduplicated = proposed[:target_clusters]

        except Exception as e:
            print(f"Error deduplicating clusters: {e}")
            deduplicated = proposed[:target_clusters]

    print(f"Final deduplicated clusters: {deduplicated}")

    # 5) assign to new fit higher level cluster
    print(
        f"Assigning {len(current_clusters)} clusters to {len(deduplicated)} higher-level clusters..."
    )
    assignments = {}
    # Clio randomly shuffles higher level cluster names to avoid order bias (what?)
    shuffled = deduplicated.copy()  # copy of higher level names
    random.shuffle(shuffled)
    higher_level_text = "\n".join([f"<cluster>{name}</cluster>" for name in shuffled])

    for cluster_id, cluster_info in current_clusters.items():
        assign_user_prompt = ASSIGN_CLUSTER_INSTR.format(higher_level_text=higher_level_text, specific_cluster=cluster_info)

        assign_assistant_prompt = """
I understand. I'll evaluate the specific cluster and assign it to
the most appropriate higher-level cluster."""

        assign_user_prompt_2 = f"""
Now, here is the specific cluster to categorize:
<specific_cluster>
Name: {cluster_info["name"]}
Description: {cluster_info["description"]}
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

            # extract answer
            if "<answer>" in content and "</answer>" in content:
                ans_start = content.find("<answer>") + 8
                ans_end = content.find("</answer>")
                assigned_cluster = content[ans_start:ans_end].strip()

                # validate assignment is in our list
                if assigned_cluster in deduplicated:
                    assignments[cluster_id] = assigned_cluster
                else:
                    # raise error or find closest match?
                    best_match = None
                    for hl_name in deduplicated:
                        if (
                            assigned_cluster.lower() in hl_name.lower()
                            or hl_name.lower() in assigned_cluster.lower()
                        ):
                            best_match = hl_name
                            break
                    assignments[cluster_id] = (
                        best_match or deduplicated[0]
                    )  # fallback, not sure if works
            else:
                assignments[cluster_id] = deduplicated[0]  # fallback
        except Exception as e:
            print(f"Error assigning cluster {cluster_id}: {e}")
            assignments[cluster_id] = deduplicated[0]

            # we now have 'assignments' list

    # 6) rename higher level clusters based on assignments- last step!!
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
            cluster_info = current_clusters[cluster_id]
            cluster_list.append(f"<cluster>({cluster_info['name']})</cluster>")
            total_size += cluster_info.get("size", cluster_info.get("total_size", 1))

        cluster_list_text = "\n".join(cluster_list)

        renamingHL_user_prompt = RENAME_CLUSTER_INSTR.format(cluster_list_text=cluster_list_text, criteria=criteria)

        renamingHL_assistant_prompt = """
Sure,  I will provide a clear, precise, and accurate summary and
name for this cluster. I will be descriptive and assume neither good nor
bad faith. Here is the summary, which I will follow with the name: <summary>"""
        try:
            response = claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                temperature=1.0,
                messages=[
                    {"role": "user", "content": renamingHL_user_prompt},
                    {"role": "user", "content": renamingHL_assistant_prompt},
                ],
            )
            content = response.content[0].text

            summary_end = content.find("</summary>")
            summary = (
                content[:summary_end].strip()
                if summary_end != -1
                else "Summary generation failed"
            )  # fallback

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
        }

        print(
            f"Level {level} Cluster {hl_id}: {name} ({len(member_cluster_ids)} sub-clusters, {total_size} total items)"
        )

    hierarchy[f"level_{level}"] = new_lvl_clusters
    hierarchy["max_level"] = level
    current_clusters = new_lvl_clusters

    print(f"Level {level} complete: {len(new_lvl_clusters)} clusters created")

    # after this loop term, we have new_lvl_clusters dict with new level clusters

print("Clustering complete! Saving Results...")

def extract_product(cluster_name):
    """Extract product from cluster name"""
    name_lower = cluster_name.lower()
    if 'langsmith' in name_lower:
        return 'LangSmith'
    elif 'langgraph platform' in name_lower:
        return 'LangGraph Platform'
    elif 'langgraph' in name_lower:
        return 'LangGraph'
    elif 'langchain' in name_lower:
        return 'LangChain'
    elif 'admin' in name_lower or 'billing' in name_lower:
        return 'Admin'
    else:
        return 'Other'

def extract_issue_type(cluster_name):
    """Extract issue type from cluster name"""
    name_lower = cluster_name.lower()
    if 'debug' in name_lower or 'error' in name_lower:
        return 'Debugging'
    elif 'setup' in name_lower or 'install' in name_lower:
        return 'Setup'
    elif 'integrat' in name_lower:
        return 'Integration'
    elif 'config' in name_lower:
        return 'Configuration'
    elif 'deploy' in name_lower:
        return 'Deployment'
    elif 'document' in name_lower:
        return 'Documentation'
    else:
        return 'Other'

output_dir = f"unthread_results-30-6/sonnet4-{L}layers-{k}base"
dataframes = save_results(
    hierarchy=hierarchy,
    example_ids=example_ids,
    cluster_labels=cluster_labels,
    summaries=summaries,
    output_dir=output_dir,
    extract_product=extract_product,
    extract_issue_type=extract_issue_type,
)
print("- level_0_clusters.csv: Base level clusters")
if hierarchy["max_level"] > 0:
    for level in range(1, hierarchy["max_level"] + 1):
        print(f"- level_{level}_clusters.csv: Level {level} clusters")

print("\nSee {output_dir} for csv files.")


# TODO - layers and k
# TODO - what visualisation/format is useful? am i generating the right csvs
