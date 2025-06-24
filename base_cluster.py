from langsmith import Client
import anthropic
from typing import Dict, Any, List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import math, time, random
from save_results import save_results
from mapping import examples_to_base_cluster, base_cluster_to_top

client = Client()
claude = anthropic.Anthropic()

# load summaries from ls
examples = list(client.list_examples(dataset_name="chat-langchain-v3-selected"))
summaries = []
example_ids = []

for example in examples:
    summaries.append(example.outputs["request"]) #i.e. request facet
    example_ids.append(example.id)

print(f"Loaded {len(summaries)} summaries")

# generate embeddings
embedder = SentenceTransformer("all-mpnet-base-v2")
embeddings = embedder.encode(summaries, show_progress_bar=True)
print(f"embeddings generated, shape: {embeddings.shape}")

# find k - can choose better method later or manually set
k = math.sqrt(len(summaries))
k = int(k)
print(f'starting to cluster, using K={k} clusters for {len(summaries)} summaries')

# cluster embeddings with k-means
kmeans = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10,
    max_iter=300
)

cluster_labels = kmeans.fit_predict(embeddings)
silhouette = silhouette_score(embeddings, cluster_labels)
print(f"silhouette score: {silhouette}") # want highest poss score

print(f'cluster labels: {cluster_labels}')

# generate descriptions for all clusters
cluster_info = {}
unique_clusters = np.unique(cluster_labels)

for cluster_id in unique_clusters:
    cluster_mask = cluster_labels == cluster_id
    cluster_summaries = [summaries[i] for i in range(len(summaries)) if cluster_mask[i]]

    #Get contrastive examples
    cluster_embeddings = embeddings[cluster_mask]
    cluster_centroid = np.mean(cluster_embeddings, axis=0)
    
    # get distances from centroid to all non-cluster points
    non_cluster_mask = ~cluster_mask
    non_cluster_embeddings = embeddings[non_cluster_mask]
    non_cluster_summaries = [summaries[i] for i in range(len(summaries)) if not cluster_mask[i]]
    
    # calculate distances to centroid
    distances = np.linalg.norm(non_cluster_embeddings - cluster_centroid, axis=1)
    
    # get closest non-cluster summaries
    n_contrastive = 50 # make this a param later
    nearest_indices = np.argsort(distances)[:n_contrastive]
    contrastive_summaries = [non_cluster_summaries[i] for i in nearest_indices]
    
    # use to generate the description
    max_summaries = 15 #arbitrary, to avoid token lims, can change later
    if len(cluster_summaries) > max_summaries:
        cluster_sample = np.random.choice(cluster_summaries, max_summaries, replace=False).tolist()  
    else:
        cluster_sample = cluster_summaries 

    if len(contrastive_summaries) > max_summaries:
        contrastive_sample = np.random.choice(contrastive_summaries, max_summaries, replace=False).tolist()
    else:
        contrastive_sample = contrastive_summaries

    cluster_sample = "\n".join(cluster_sample) # list to str
    contrastive_sample = "\n".join(contrastive_sample)

    prompt = f"""You are tasked with summarizing a group of related statements into a short, precise, and accurate description and name. Your goal is to create a concise summary that captures the essence of these statements and distinguishes them from other similar groups of statements.

Summarize all the statements into a clear, precise, two-sentence description in the past tense. Your summary should be specific to this group and distinguish it from the contrastive answers of the other groups.

After creating the summary, generate a short name for the group of statements. This name should be at most ten words long (perhaps less) and be specific but also reflective of most of the statements (rather than reflecting only one or two).

The name should distinguish this group from the contrastive examples. For instance, "Write fantasy sexual roleplay with octopi and monsters", "Generate blog spam for gambling websites", or "Assist with high school math homework" would be better and more actionable than general terms like "Write erotic content" or "Help with homework". Be as descriptive as possible and assume neither good nor bad faith. Do not hesitate to identify and describe socially harmful or sensitive topics specifically; specificity is necessary for monitoring.

Present your output in the following format:
<summary> [Insert your two-sentence summary here] </summary>
<name> [Insert your generated short name here] </name>

The names you propose must follow these requirements:
<criteria>The cluster name should be a sentence in the imperative that captures the user's request. For example, 'Brainstorm ideas for a birthday party' or 'Help me find a new job.'</criteria>

Below are the related statements:
<answers>
{cluster_sample}
</answers>

For context, here are statements from nearby groups that are NOT part of the group you're summarizing:
<contrastive_answers>
{contrastive_sample}
</contrastive_answers>

Do not elaborate beyond what you say in the tags. Remember to analyze both the statements and the 
contrastive statements carefully to ensure your summary and name accurately represent the specific 
group while distinguishing it from others."""
    
    starter = f"""Sure, I will provide a clear, precise, and accurate summary and name for
this cluster. I will be descriptive and assume neither good nor bad faith. Here
is the summary, which I will follow with the name: <summary>""" 

    try:
        response = claude.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            temperature=1.0,
            messages=[
                {
                    "role": "user", 
                    "content": prompt},
                {
                    "role": "assistant", 
                    "content": starter}
                ]
        )

        content = response.content[0].text
        summary_end = content.find('</summary>')
        summary = content[:summary_end].strip() if summary_end != -1 else "Summary generation failed"

        name_start = content.find('<name>') + 6
        name_end = content.find('</name>')
        name = content[name_start:name_end].strip()

    except Exception as e:
        print(f"Error: {e}")
        name = f"Cluster {cluster_id}"
        summary = "Error generating description"

    cluster_info[cluster_id] = {
        'name': name,
        'description': summary,
        'size': len(cluster_summaries),
        'summaries': cluster_summaries[:10] # just first 10 for inspection
    }

    print(f"Cluster {cluster_id}: {name} ({len(cluster_summaries)} items)")
    time.sleep(0.5)

# cluster loop complete, now preparing dataset updates
print("updating dataset with cluster assignments...")
updates = []
for i, (example_id, cluster_id) in enumerate(zip(example_ids, cluster_labels)):
    cluster_name = cluster_info[cluster_id]['name']
    example = examples[i] 
    
    update = {
        "id": example_id,
        "metadata": example.metadata,
        "inputs": example.inputs,
        "outputs": {
            "request": example.outputs["request"],  # keep existing request, can change this later when generating csvs
            "cluster_id": int(cluster_id),
            "cluster_name": cluster_name,
        }
    }
    updates.append(update)

# update all 
client.update_examples(dataset_name="chat-langchain-v3-selected", updates=updates)
print("Dataset updated!")

# print results
print(f"\nResults:")
for cluster_id, info in cluster_info.items():
    print(f"\nCluster {cluster_id}: {info['name']}")
    print(f"Description: {info['description']}")
    print(f"Size: {info['size']} conversations")
    print(f"Sample summaries: {info['summaries'][:3]}")

print(f"base cluster complete!")


# starting hierarchical clustering (G7)
# load base cluster names from ls

hierarchy = {
        'level_0': cluster_info,
        'max_level': 0
}
current_clusters = cluster_info
n_base = len(current_clusters)

# various variable declarations to change later/make configurable
ktop = 4 # desired no. of top level clusters, 10 in clio
L = 2 # no of levels, 3 in clio
avg_clusters_per_neighborhood = 8 # 40 in clio
m_nearest = 3 #nearest clusters, 10 in clio

# clio's geometric progression algo for no. of clusters at each level
if L == 2:
    level_sizes = [ktop] #for 2 levels, just base to top
else: 
    ratio = (ktop/n_base) ** (1/ (L-1) )
    level_sizes = []
    for level in range(1, L):
        n_level = int(n_base * (ratio ** level))
        level_sizes.append(max(2, n_level)) # make sure atleast 2 clusters
    level_sizes.append(ktop)

print(f"planned hierarchy sizes: {[n_base] + level_sizes}")

#build hierarchy lvl by lvl
for level in range(1,L):
    if len(current_clusters) <= level_sizes[level-1]:
        print(f"stopping at level {level-1} bc only {len(current_clusters)} left")
        break

    print(f'creating level {level}, targetting {level_sizes[level-1]} clusters')

    # 1) embed clusters
    cluster_names_descriptions = []
    cluster_ids = []

    for cluster_id, info in current_clusters.items():
        #combine name and description for embedding
        text = f"{info['name']}: {info['description']}"
        cluster_names_descriptions.append(text)
        cluster_ids.append(cluster_id)

    print(f"embedding {len(cluster_ids)} cluster descriptions...")
    cluster_embeddings = embedder.encode(cluster_names_descriptions, show_progress_bar=True)


    # 2) generate neighbourhoods using k-means
    k_neighborhoods = max(1, len(current_clusters)//avg_clusters_per_neighborhood)
    print(f'making {k_neighborhoods} neighborhoods with approx {avg_clusters_per_neighborhood} clusters per neighborhood')

    kmeans = KMeans(
        n_clusters = k_neighborhoods,
        random_state = 42,
        n_init=10,
        max_iter=300
    )
    
    neighborhood_labels = kmeans.fit_predict(cluster_embeddings)

    # 3) now propose new clusters for each neighborhood
    proposed = []
    target_clusters = level_sizes[level-1]
    clusters_per_neighborhood = max(1, target_clusters // k_neighborhoods) 

    for neighborhood_id in range(k_neighborhoods):
        neighborhood_mask = neighborhood_labels == neighborhood_id
        neighborhood_cluster_ids = [cluster_ids[i] for i in range(len(cluster_ids)) if neighborhood_mask[i]]
        neighborhood_clusters = {cid: current_clusters[cid] for cid in neighborhood_cluster_ids}
        
        # find nearest m clusters outside this nbh for contrastive
        neighborhood_embeddings = cluster_embeddings[neighborhood_mask]
        neighborhood_centroid = np.mean(neighborhood_embeddings, axis=0)

        outside_mask = ~neighborhood_mask
        if np.any(outside_mask):
            outside_embeddings = cluster_embeddings[outside_mask]
            distances = np.linalg.norm(outside_embeddings - neighborhood_centroid, axis=1)
            nearest_indices = np.argsort(distances)[:m_nearest] #was manually set to 3
            outside_cluster_ids = [cluster_ids[i] for i in range(len(cluster_ids)) if outside_mask[i]]
            nearest_outside_ids = [outside_cluster_ids[i] for i in nearest_indices]
            nearest_outside_clusters = {cid: current_clusters[cid] for cid in nearest_outside_ids}
        else: #nothing outside this cluster - updated min clusteres to 2 so shouldn't happen
            nearest_outside_clusters = {}

        # now actually ready to propose clusters
        # nbh_proposed = propose_clusters_for_neighborhood

        # build cluster list for prompt
        cluster_list = []
        for cluster_id, info in neighborhood_clusters.items():
            cluster_list.append(f"<cluster>{info['name']}: {info['description']}</cluster>")

        cluster_list_text = "\n".join(cluster_list)

        criteria = """ The cluster name should be a sentence in the
imperative that captures the user's request. For example, 'Brainstorm ideas for a birthday party' or
'Help me find a new job.' """

        proposing_user_prompt = f"""
        You are tasked with creating higher-level cluster names based on a
given list of clusters and their descriptions. Your goal is to come up
with broader categories that could encompass one or more of the provided
clusters.

First, review the list of clusters and their descriptions:
<cluster_list>
{cluster_list_text}
</cluster_list>

Your task is to create roughly {clusters_per_neighborhood} higher-level cluster names
that could potentially include one or more of the provided clusters.
These higher-level clusters should represent broader categories or
themes that emerge from the given clusters, while remaining as specific
as possible. If there are many clusters with a specific theme, ensure
that the higher-level cluster name remains the maximum level of
specificity. You are helping to organize user behavior data in order to
improve safety, monitoring, and observability. You can generate more or
less than {clusters_per_neighborhood} names if you feel that more or fewer are
appropriate and accurately capture the clusters. You should output at
least {int(0.5 * clusters_per_neighborhood)} and at most {int(1.5 * clusters_per_neighborhood)}
names, with {clusters_per_neighborhood} as a target.

Guidelines for creating higher-level cluster names:
1. Analyze the themes, topics, or characteristics common to multiple clusters.
2.  Create names that are specific enough to be meaningful, but not so
specific that they can't meaningfully represent many different clusters.
Avoid overly general or vague terms, and do not hesitate to describe
socially harmful or sensitive topics (in fact, clusters that clearly
describe harmful behavior are slightly preferred); specificity is
necessary for observability and enforcement.
3.  Ensure that the higher-level cluster names are distinct from one another.
4.  Use clear, concise, and descriptive language for the cluster names.
Assume neither good nor bad faith for the content in the clusters.

The names you propose must follow these requirements:
<criteria>{criteria}</criteria>

Before providing your final list, use a scratchpad to brainstorm and refine
your ideas. Think about the relationships between the given clusters and
potential overarching themes.

<scratchpad>
[Use this space to analyze the clusters, identify common themes, and
brainstorm potential higher-level cluster names. Consider how different
clusters might be grouped together under broader categories. No longer
than a paragraph or two.]
</scratchpad>

Now, provide your list of roughly {clusters_per_neighborhood} higher-level cluster names. Present your answer in 
the following format:
<answer>
1. [First higher-level cluster name]
2. [Second higher-level cluster name]
3. [Third higher-level cluster name]
...
{clusters_per_neighborhood}. [Last higher-level cluster name]
</answer>

Focus on creating meaningful, distinct, and precise (but not overly specific
) higher-level cluster names that could encompass multiple sub-clusters.
"""
        #criteria should be made a variable
        # above, figure out where we need variables and where we don't
        # later we should put all these prompts in the prompt hub

        proposing_assistant_prompt = f""" I understand. I'll evaluate the clusters and provide higher-level
cluster names that could encompass multiple sub-clusters.

<scratchpad>"""
        
        try:
            response = claude.messages.create(
                model="claude-3-5-sonnet-20241022", #update?
                max_tokens=1000,
                temperature=1.0, # in clio
                messages=[
                    {
                        "role":"user",
                        "content": proposing_user_prompt
                    },
                    { 
                        "role": "assistant",
                        "content": proposing_assistant_prompt
                    }
                ]
            )
            content = response.content[0].text
            print(f"Claude response for neighborhood {neighborhood_id}:")
            print(content[:200] + "..." if len(content) > 200 else content)

            proposed_names = []
            #extract answer section
            if '<answer>' in content and '</answer>' in content:
                ans_start = content.find('<answer>')
                ans_end = content.find('</answer>')
                ans_text = content[ans_start:ans_end].strip()

                # we should have a numbered list to parse, for ex:
                # 1. [First higher-level cluster name]
                # 2. [Second higher-level cluster name]
                # 3. [Third higher-level cluster name]
                for line in ans_text.split('\n'):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-')): # not sure if numbering or bullet
                        name = line.split('.',1)[1].strip() if '.' in line else line.strip()
                        if name.startswith('[') and name.endswith(']'):
                            name=name[1:-1]
                        proposed_names.append(name) # we now have proposed_names
            if not proposed_names:
                print(f"Warning: No names extracted from Claude response for neighborhood {neighborhood_id}")
                # Create fallback names based on the actual clusters in this neighborhood
                if len(neighborhood_clusters) > 0:
                    sample_names = [info['name'] for info in list(neighborhood_clusters.values())[:2]]
                    proposed_names = [f"Handle {' and '.join(sample_names[:2])} related requests"]
                else:
                    proposed_names = [f"Cluster Group {neighborhood_id}"]

            proposed.extend(proposed_names)
            print(f"Neighborhood {neighborhood_id} proposed: {proposed_names}")

        except Exception as e:
            print(f"Error, could not propose clusters for neighborhood {neighborhood_id}")
                        
    

    # 4) deduplicate across neighborhoods
    print(f"Deduplicating {len(proposed)} proposed clusters")
    if len(proposed) == 0:
        print("ERROR: No clusters were proposed!")
    if len(proposed) <= clusters_per_neighborhood:
        deduplicated = proposed #
    else:
        cluster_text = "\n".join([f"<cluster>{name}</cluster>" for name in proposed]) #for prompt
        deduplicating_user_prompt = f""" You are tasked with deduplicating a list of cluster names into a
smaller set of distinct cluster names. Your goal is to create
approximately {clusters_per_neighborhood} relatively distinct clusters that best
represent the original list. You are helping to organize user behavior
data in order to improve safety, monitoring, and observability. Here are
the inputs:

<cluster_names>
{cluster_text}
</cluster_names>

Number of distinct clusters to create: approximately {clusters_per_neighborhood}

Follow these steps to complete the task:
1. Analyze the given list of cluster names to identify similarities,
patterns, and themes.
2. Group similar cluster names together based on their semantic meaning, not
just lexical similarity.
3. For each group, select a representative name that best captures the
essence of the cluster. This can be one of the original names or a new
name that summarizes the group effectively. Do not just pick the most
vague or generic name.
4. Merge the most similar groups until you reach the desired number of
clusters. Maintain as much specificity as possible while merging.
6. Ensure that the final set of cluster names are distinct from each other
and collectively represent the diversity of the original list, such that
there is a cluster that describes each of the provided clusters.
7. If you create new names for any clusters, make sure they are clear,
concise, and reflective of the contents they represent.
8. You do not need to come up with exactly {clusters_per_neighborhood} names, but aim
for no less than {int(clusters_per_neighborhood * 0.5)} and no more than {int(
clusters_per_neighborhood * 1.5)}. Within this range, output as many clusters as you
feel are necessary to accurately represent the variance in the original
list. Avoid outputting duplicate or near-duplicate clusters.
9. Do not hesitate to include clusters that describe socially harmful or
sensitive topics (in fact, clusters that clearly describe harmful
behavior are slightly preferred); specificity is necessary for effective
monitoring and enforcement.
10. Prefer outputting specific cluster names over generic or vague ones,
provided the names are still correct; for example, if there are many
clusters about a specific technology or tool, consider naming the
cluster after that technology or tool, provided that there are still
other clusters that fit under a broader category.

The names you propose must follow these requirements:

<criteria>{criteria}</criteria>

Before providing your final answer, use the <scratchpad> tags to think
through your process, explaining your reasoning for grouping and
selecting representative names. Spend no more than a few paragraphs in
your scratchpad.

Present your final answer in the following format: 

<answer>
1. [First cluster name]
2. [Second cluster name]
3. [Third cluster name]
...
N. [Nth cluster name]
</answer>

Remember, your goal is to create approximately {clusters_per_neighborhood} relatively
distinct cluster names that best represent the original list. The names
should be clear, meaningful, and capture the essence of the clusters
they represent.
"""
        deduplicating_assistant_prompt = f"""
        I understand. I'll deduplicate the cluster names into approximately {clusters_per_neighborhood} names.
        <scratchpad>"""
        try:
            response = claude.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=1000,
                temperature=1.0,
                messages=[
                    {
                        "role": "user",
                        "content": deduplicating_user_prompt
                    },
                    {
                        "role": "assistant",
                        "content": deduplicating_assistant_prompt
                    }
                ]
            )
            content = response.content[0].text

            deduplicated = []
            #extract answer section
            if "<answer>" in content and "</answer>" in content:
                ans_start = content.find("<answer>") + 8
                ans_end = content.find("</answer>")
                ans_text = content[ans_start:ans_end].strip()

                # numbered list like above to parse
                for line in ans_text.split('\n'):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-')):
                        # remove and clean up
                        name = line.split('.',1)[1].strip() if '.' in line else line.strip()
                        if name.startswith('[') and name.endswith(']'): #can I assume it always does or no?
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
    print(f"Assigning {len(current_clusters)} clusters to {len(deduplicated)} higher-level clusters...")
    assignments = {}
        # Clio randomly shuffles higher level cluster names to avoid order bias (what?)
    shuffled = deduplicated.copy() #copy of higher level names
    random.shuffle(shuffled)
    higher_level_text = '\n'.join([f'<cluster>{name}</cluster>' for name in shuffled])
        
    for cluster_id, cluster_info in current_clusters.items(): 
        assign_user_prompt = f"""
You are tasked with categorizing a specific cluster into one of the provided
higher-level clusters for observability, monitoring, and content
moderation. Your goal is to determine which higher-level cluster best
fits the given specific cluster based on its name and description. You
are helping to organize user behavior data in order to improve safety,
monitoring, and observability.
First, carefully review the following list of higher-level clusters (
hierarchy denoted by dashes):

<higher_level_clusters>
{higher_level_text}
<higher_level_clusters>

To categorize the specific cluster:
1. Analyze the name and description of the specific cluster.
2.  Consider the key characteristics, themes, or subject matter of the
specific cluster.
3.  Compare these elements to the higher-level clusters provided.
4. Determine which higher-level cluster best encompasses the specific
cluster. You MUST assign the specific cluster to the best higher-level
cluster, even if multiple higher-level clusters could be considered.
5. Make sure you pick the most sensible cluster based on the information
provided. For example, don't assign a cluster about "Machine Learning"
to a higher-level cluster about "Social Media" just because both involve
technology, and don't assign a cluster about "Online Harassment" to a
higher-level cluster about "Technology" just because both involve online
platforms. Be specific and accurate in your categorization.

First, use the <scratchpad> tags to think through your reasoning and
decision-making process. Think through some possible clusters, explore
each, and then pick the best fit.

<scratchpad>
In a few brief sentences, think step by step, explain your reasoning, and
finally determine which higher-level cluster is the best fit for the
specific cluster.
</scratchpad>

Then, provide your answer in the following format:
<answer>
[Full name of the chosen cluster, exactly as listed in the higher-level
clusters above, without enclosing <cluster> tags]
</answer>
"""
        assign_assistant_prompt = f"""
I understand. I'll evaluate the specific cluster and assign it to
the most appropriate higher-level cluster."""
            
        assign_user_prompt_2 = f"""
Now, here is the specific cluster to categorize:
<specific_cluster>
Name: {cluster_info['name']}
Description: {cluster_info['description']}
</specific_cluster>

Based on this information, determine the most appropriate higher-level
cluster and provide your answer as instructed."""
        assign_assistant_prompt_2 = f""" 
Thank you, I will reflect on the cluster and categorize it most
appropriately, which will help with safety, moderation, and
observability.

<scratchpad>"""
        try:
            response = claude.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=500,
            temperature=1.0,
            messages = [
                {
                    "role":"user",
                    "content": assign_user_prompt
                },
                {
                    "role": "assistant",
                    "content": assign_assistant_prompt
                },
                {
                    "role": "user",
                    "content": assign_user_prompt_2
                },
                {
                    "role": "assistant",
                    "content": assign_assistant_prompt_2
                    }
                ]
            )
            content = response.content[0].text

            #extract answer
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
                        if assigned_cluster.lower() in hl_name.lower() or hl_name.lower() in assigned_cluster.lower():
                            best_match = hl_name
                            break
                    assignments[cluster_id] = best_match or deduplicated[0] #fallback, not sure if works 
            else:
                assignments[cluster_id] = deduplicated[0] #fallback
        except Exception as e:
            print(f"Error assigning cluster {cluster_id}: {e}")
            assignments[cluster_id] = deduplicated[0]

            # we now have 'assignments' list

    # 6) rename higher level clusters based on assignments- last step!! 
    print("Renaming higher-level clusters based on assignments...")
    new_lvl_clusters ={}

    # group clusters by their assigned HL cluster
    cluster_groups = {}
    for cluster_id, assigned_hl in assignments.items():
        if assigned_hl not in cluster_groups:
            cluster_groups[assigned_hl] = []
        cluster_groups[assigned_hl].append(cluster_id)

    for hl_id, (hl_name, member_cluster_ids) in enumerate(cluster_groups.items()):
        # building list of member clusters for prompt
        cluster_list = [] #changed from members
        total_size = 0
        for cluster_id in member_cluster_ids:
            cluster_info = current_clusters[cluster_id]
            cluster_list.append(f"<cluster>({cluster_info['name']})</cluster>")
            total_size += cluster_info.get('size', cluster_info.get('total_size', 1))

        cluster_list_text = '\n'.join(cluster_list) 

            
        renamingHL_user_prompt = f"""
You are tasked with summarizing a group of related cluster names into
a short, precise, and accurate overall description and name. Your goal
is to create a concise summary that captures the essence of these
clusters.

Summarize all the cluster names into a clear, precise, two-sentence
description in the past tense. Your summary should be specific to this
cluster.

After creating the summary, generate a short name for the cluster. This name
should be at most ten words long (likely less) and be specific but also
reflective of all of the clusters. For instance, "Write fantasy sexual
roleplay with octopi and monsters", "Generate blog spam for gambling
websites", or "Assist with high school math homework" would be better and 
more actionable than general terms like "Write erotic content" or "Help with homework". 
Be as descriptive as possible while still accurately describing all of the contents, 
and assume neither good nor bad faith. Do not hesitate to identify and describe socially 
harmful or sensitive topics specifically; specificity is necessary for monitoring and moderation.

Present your output in the following format:
<summary> [Insert your two-sentence summary here] </summary>
<name> [Insert your generated short name here, with no period or trailing
punctuation] </name>

The name you choose must follow these requirements:
<criteria>{criteria}</criteria>
Below are the related statements:
<answers>
{cluster_list_text}
</answers>
Do not elaborate beyond what you say in the tags. Ensure your summary and
name accurately represent the clusters.
"""

        renamingHL_assistant_prompt = f"""
Sure,  I will provide a clear, precise, and accurate summary and
name for this cluster. I will be descriptive and assume neither good nor
bad faith. Here is the summary, which I will follow with the name: <summary>"""
        try:
            response = claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=1.0,
                messages=[
                    {
                        "role":"user",
                        "content": renamingHL_user_prompt
                    },
                    {
                        "role":"user",
                        "content": renamingHL_assistant_prompt
                    }
                ]
            )
            content = response.content[0].text

            summary_end = content.find("</summary>")
            summary = content[:summary_end].strip() if summary_end != -1 else "Summary generation failed" #fallback

            name_start = content.find("<name>") + 6
            name_end = content.find("</name>")
            name = content[name_start:name_end].strip() if name_start != -1 and name_end != -1 else f"Level {level} Cluster {hl_id}"

        except Exception as e:
            print(f'Error renaming cluster {hl_name}: {e}')
            name = f"Level {level} Cluster {hl_id}"
            summary = "Summary generation failed"

        new_lvl_clusters[hl_id] = {
            "name": name,
            "description": summary,
            "member_clusters": member_cluster_ids,
            "total_size": total_size,
            "size": len(member_cluster_ids)
        }

        print(f"Level {level} Cluster {hl_id}: {name} ({len(member_cluster_ids)} sub-clusters, {total_size} total items)")

    hierarchy[f'level_{level}'] = new_lvl_clusters
    hierarchy['max_level'] = level
    current_clusters = new_lvl_clusters

    print(f"Level {level} complete: {len(new_lvl_clusters)} clusters created")

        # after this loop term, we have new_lvl_clusters dict with new level clusters

print("Clustering complete! Saving Results...")
dataframes = save_results(
    hierarchy=hierarchy,
    example_ids=example_ids, 
    cluster_labels=cluster_labels,
    summaries=summaries,
    output_dir="clustering_results"
)
print(f"- level_0_clusters.csv: Base level clusters")
if hierarchy['max_level'] > 0:
    for level in range(1, hierarchy['max_level'] + 1):
        print(f"- level_{level}_clusters.csv: Level {level} clusters")

print(f"\nSee /clustering_results for csv files.")

