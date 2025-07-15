import pandas as pd
import os
import glob
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from logging import getLogger
import warnings
import asyncio
from langsmith import Client
from open_clio.prompts import PARTITION_AND_SUMMARIZE, BASE_CLUSTER

logger = getLogger(__file__)

llm = init_chat_model(
    "anthropic:claude-sonnet-4-20250514",
    temperature=0.4, # ?
    max_tokens = 500
)


def load_examples(dataset_name, sample=None):
    client = Client()
    examples = list(client.list_examples(dataset_name=dataset_name, limit=sample))
    print(f"Loaded {len(examples)} examples from {dataset_name}")
    return examples

async def process_single_examples(example, existing_data, expected_partition=None):
    # Choose partition and summarise each ex
    partition_result = await determine_partition_and_summarize(example, existing_data)
    if not partition_result:
        warnings.warn(f"Failed to determine partition for {example.id}")
        return None
    
    partition = partition_result['partition']
    summary = partition_result['summary']
    
    # Assign to base cluster within that partition
    level_0_result = await assign_to_base_cluster(example, existing_data, partition, summary)
    if not level_0_result:
        warnings.warn(f"Failed to assign level 0 for {example.id}")
        return None
    
    combined_result = {
        'summary': summary,
        'partition': partition,
        'level_0_cluster_id': level_0_result['level_0_cluster_id'],
        'level_0_cluster_name': level_0_result['level_0_cluster_name']
    }
    
    # Find higher levels using level 0 assignment
    full_assignments = await assign_higher_levels(combined_result, existing_data, partition)

    # Fmt full_example (from generate.py)
    full_example = ""
    if example.inputs:
        if isinstance(example.inputs, dict):
            input_parts = []
            for key, value in example.inputs.items():
                if isinstance(value, str) and value.strip():
                    input_parts.append(f"{key}: {value}")
            full_example = "\n".join(input_parts)
        elif isinstance(example.inputs, str):
            full_example = example.inputs
        else:
            full_example = str(example.inputs)
    else:
        full_example = summary  # fallback

    # Fmt for extend_results
    assignment_result = {
        'example_id': example.id,
        'summary': summary,
        'partition': partition,
        'full_example': full_example,
        'assignments': full_assignments
    }
    print(f"Successfully assigned example {example.id} (partition: {partition}, level0: {level_0_result['level_0_cluster_name']}, level1+: see csvs)")
    return assignment_result

def load_hierarchy(save_path):
    combined_csv_path = f"{save_path}/combined.csv"
    
    if not os.path.exists(combined_csv_path):
        print(f"Error: Cluster data not found at {combined_csv_path}")
        print("Please run generate.py first to create the initial clusters before extending.")
        exit(1)
    
    combined_df = pd.read_csv(combined_csv_path)
    
    level_clusters = {}
    pattern = os.path.join(save_path, "level_*_clusters.csv")
    cluster_files = glob.glob(pattern)
    
    for file_path in sorted(cluster_files):
        # Save each as level_x.csv
        filename = os.path.basename(file_path)
        level_name = filename.replace('_clusters.csv', '')  # e.g., "level_0"
        df = pd.read_csv(file_path)
        level_clusters[level_name] = df
        print(f"Loaded {len(df)} clusters from {filename}")
    
    # Get partitions
    partitions = combined_df["partition"].unique().tolist()
    
    # Calculate max_levels and cluster_names for this specific partition
    max_levels_by_partition = {}
    cluster_names_by_partition = {}
    
    for partition in partitions:
        partition_examples = combined_df[combined_df["partition"] == partition]
        
        # max level
        max_level = 0
        level_columns = [col for col in partition_examples.columns if col.startswith('level_') and col.endswith('_id')]
        for col in level_columns:
            if partition_examples[col].notna().any():
                level_num = int(col.split('_')[1])
                max_level = max(max_level, level_num)
        
        max_levels_by_partition[partition] = max_level
        
        # cluster names
        cluster_names_by_partition[partition] = {}
        for level in range(max_level + 1):
            level_key = f'level_{level}'
            
            if level_key not in level_clusters:
                continue
            
            clusters_df = level_clusters[level_key]
            partition_clusters = clusters_df[clusters_df['partition'] == partition]
            
            cluster_names_by_partition[partition][level_key] = partition_clusters[['cluster_id', 'name']].to_dict('records')
    
    print(f"\nFound {len(partitions)} partitions: {partitions}")
    for partition, max_level in max_levels_by_partition.items():
        print(f"  {partition}: levels 0-{max_level}")
    
    return {
        'combined_df': combined_df,
        'level_clusters': level_clusters,
        'partitions': partitions,
        'max_levels_by_partition': max_levels_by_partition,
        'cluster_names_by_partition': cluster_names_by_partition  # just ID + name per partition/level
    }

async def determine_partition_and_summarize(example, existing_data, partitions=None):
    """Determine which partition the example belongs to and generate a summary."""
    
    if partitions is None:
        available_partitions = existing_data['partitions']
    else:
        available_partitions = list(partitions.keys()) if isinstance(partitions, dict) else partitions
    prompt = PARTITION_AND_SUMMARIZE.format(
            available_partitions=available_partitions,
            example=example
        )

    class PartitionResponseFormatter(BaseModel):
        summary: str = Field(
            description="A structured summary of the support conversation that captures the main task, request, or purpose. Focus on what the user is asking for, be specific about the subject matter or domain, and include context about the purpose or use case when relevant. Do NOT include phrases like 'User requested' or 'I understand' - start directly with the action/task."
        )
        partition: str = Field(
            description=f"The partition this example belongs to. Must be one of: {available_partitions}"
        )
    
    structured_llm = llm.with_structured_output(PartitionResponseFormatter)
    
    try:

        response = await structured_llm.ainvoke(prompt)
        
        summary = response.summary
        partition = response.partition
        
        # make sure it's a valid partition
        if partition not in available_partitions:
            logger.error(f"Invalid partition '{partition}' returned for example {example.id}. Available: {available_partitions}")
            return None
            
    except Exception as e:
        logger.error(f"Error determining partition for example {example.id}: {e}")
        return None
    
    return {
        'summary': summary,
        'partition': partition
    }

async def assign_to_base_cluster(example, existing_data, partition, summary):
    """Assign example to a base cluster within the specified partition."""
    
    level_0_clusters = existing_data['cluster_names_by_partition'][partition]['level_0']
    cluster_options = ""
    for i, cluster in enumerate(level_0_clusters):
        cluster_options += f"{i+1}. {cluster['name']}\n"
    
    prompt = BASE_CLUSTER.format(
        summary=summary,
        partition=partition,
        cluster_options=cluster_options,
        example=example
    )
    class ClusterResponseFormatter(BaseModel):
        level_0_cluster_name: str = Field(
            description=f"The exact name of the level 0 cluster that the example best belongs to. Must be one of: {[c['name'] for c in level_0_clusters]}"
        )
    
    structured_llm = llm.with_structured_output(ClusterResponseFormatter)
    example_text = str(example.inputs)
    
    try:
        response = await structured_llm.ainvoke(prompt)
        level_0_cluster_name = response.level_0_cluster_name
        
        # Look up the UUID from the cluster name 
        level_0_cluster_id = None
        for cluster in level_0_clusters:
            if cluster['name'] == level_0_cluster_name:
                level_0_cluster_id = cluster['cluster_id']
                print(f"Assigned example {example.id} to level 0 cluster {level_0_cluster_name} (ID: {level_0_cluster_id})")
                break
        
        if level_0_cluster_id is None:
            logger.error(f"Could not find cluster ID for name '{level_0_cluster_name}' in partition '{partition}'")
            return None
            
    except Exception as e:
        logger.error(f"Error assigning to base cluster for example {example.id}: {e}")
        return None

    return {
        'level_0_cluster_id': level_0_cluster_id,
        'level_0_cluster_name': level_0_cluster_name
    }

async def assign_higher_levels(level_0_assignment, partition_clusters, partition_name="Default"):

    assignments = {
        'level_0': {
            'cluster_id': level_0_assignment['level_0_cluster_id'],
            'cluster_name': level_0_assignment['level_0_cluster_name'],
        }
    }

    max_level = partition_clusters['max_levels_by_partition'][partition_name]
    current_assignment_id = level_0_assignment['level_0_cluster_id']
    combined_df = partition_clusters['combined_df'] # is this right

    # for each higher level, find available options and assign
    for level in range(1, max_level+1):
        level_key = f'level_{level}'
        level_id_col = f'{level_key}_id'
        
        # Handling key error: for level 1 where previous level is base_cluster_id
        if level == 1:
            previous_level_col = 'base_cluster_id'
        else:
            previous_level_col = f'level_{level-1}_id'

        # current level = n
        # find which n+1 clusters contain this level n cluster
        # by looking in combined_df for examples with our current level n cluster and looking at what n+1 clusters they're in
        examples_with_this_level_n = combined_df[combined_df[previous_level_col] == current_assignment_id]
        if len(examples_with_this_level_n) == 0:
            warnings.warn(f"No examples found for level {level} cluster {current_assignment_id}")
            break

        # unique level n+1 cluster ids that contain current level n
        valid_cluster_ids = examples_with_this_level_n[level_id_col].dropna().unique()

        # get all level n+1 clusters that contain these examples
        available_clusters = partition_clusters['cluster_names_by_partition'][partition_name][level_key]
        valid_clusters = [c for c in available_clusters if c['cluster_id'] in valid_cluster_ids]

        assert len(valid_clusters) == 1, f"Multiple level {level_key} clusters found for {current_assignment_id}: {valid_clusters}"
        chosen_cluster = valid_clusters[0]
        print(f"Assigning level {level} cluster {chosen_cluster['name']} to {current_assignment_id} (only option)")
        
        assignments[level_key] = {
            'cluster_id': chosen_cluster['cluster_id'],
            'cluster_name': chosen_cluster['name']
        }

        current_assignment_id = chosen_cluster['cluster_id'] # unecessary?

    return assignments
    

# before extend_results - check assignment logic works
def extend_results(new_assignments, save_path):
    """
    Updates combined.csv and level_{x}_clusters.csv files
    """
    #append not overwrite
    print(f"Extending results with {len(new_assignments)} new assignments, in {save_path}")
    prev_combined = pd.read_csv(os.path.join(save_path, "combined.csv"))

    logger.info(f"Previous combined.csv has {len(prev_combined)} examples/rows")

    new_rows = []  # will add to prev_combined
    cluster_updates = {}

    for assignment in new_assignments:
        example_id = assignment['example_id']
        summary = assignment['summary']
        partition = assignment['partition']
        full_example = assignment.get('full_example', summary)  # summary as fallback

        # Match existing combined.csv structure
        row_data = {
            'example_id': example_id,
            'full_example': full_example,
            'summary': summary,
            'partition': partition,
        }

        # Add cluster assingnments for each level
        assignments_dict = assignment['assignments']  # from assign_higher_levels

        # And base cluster info
        if 'level_0' in assignments_dict:
            row_data['base_cluster_id'] = assignments_dict['level_0']['cluster_id']
            row_data['base_cluster_name'] = assignments_dict['level_0']['cluster_name']

            # Track cluster size update
            cluster_id = assignments_dict['level_0']['cluster_id']
            if cluster_id not in cluster_updates:
                cluster_updates[cluster_id] = {'level': 0, 'count': 0}
            cluster_updates[cluster_id]['count'] += 1
        
        # Add intermediate and top cluster levels
        max_level = max([int(k.split('_')[1]) for k in assignments_dict.keys() if k.startswith('level_')])
        for level in range(1, max_level+1):
            level_key = f'level_{level}'
            if level_key in assignments_dict:
                row_data[f'{level_key}_id'] = assignments_dict[level_key]['cluster_id']
                row_data[f'{level_key}_name'] = assignments_dict[level_key]['cluster_name']

                # Track cluster size update for these too
                cluster_id = assignments_dict[level_key]['cluster_id']
                if cluster_id not in cluster_updates:
                    cluster_updates[cluster_id] = {'level': level, 'count': 0}
                cluster_updates[cluster_id]['count'] += 1

        # Cp top cluster info from highest level
        if max_level >= 0: #does this work if we ONLY have base clusters
            top_level_key = f"level_{max_level}"
            if top_level_key in assignments_dict:
                row_data['top_cluster_id'] = assignments_dict[top_level_key]['cluster_id']
                row_data['top_cluster_name'] = assignments_dict[top_level_key]['cluster_name']

        new_rows.append(row_data)
    
    # Convert to df with columns as prev_combined df
    new_df = pd.DataFrame(new_rows)
    for col in prev_combined.columns:
        if col not in new_df.columns:
            new_df[col] = None

    new_df = new_df[prev_combined.columns]

    # Combine, save
    updated_combined = pd.concat([prev_combined, new_df], ignore_index=True)
    updated_combined.to_csv(os.path.join(save_path, "combined.csv"), index=False)
    print(f"Updated combined.csv: added {len(new_rows)} new rows, total {len(updated_combined)} rows")

    # Update cluster sizes
    update_cluster_files(cluster_updates, save_path)


def update_cluster_files(cluster_updates, save_path):
    """
    Update the size counts in level_x_cluster.csv files
    """
    print("Updating cluster sizes...")

    # group updates by level - separate csvs
    updates_by_level = {}
    for cluster_id, info in cluster_updates.items():
        level = info['level']
        count = info['count']

        if level not in updates_by_level:
            updates_by_level[level] = {}
        updates_by_level[level][cluster_id] = count

    for level, cluster_counts in updates_by_level.items():
        # read from original level file
        original_level_file = os.path.join(save_path, f"level_{level}_clusters.csv")
        updated_level_file = os.path.join(save_path, f"level_{level}_clusters.csv")

        if not os.path.exists(original_level_file):
            warnings.warn(f"Level {level} cluster file not found: {original_level_file}, cannot update {len(cluster_counts)} level {level} clusters")
            continue

        df = pd.read_csv(original_level_file)
        
        for cluster_id, additional_count in cluster_counts.items():
            # find the row
            mask = df['cluster_id'] == cluster_id

            if mask.any():  # update size
                if level == 0:
                    # For level 0 clusters: size = number of examples
                    current_size = df.loc[mask, 'size'].iloc[0]
                    new_size = current_size + additional_count
                    df.loc[mask, 'size'] = new_size
                    print(f"  Updated Level {level} cluster {cluster_id}: {current_size} -> {new_size} examples")
                else:
                    # For higher-level clusters: 
                    # - size = number of sub-clusters (doesn't change when adding examples)
                    # - total_size = number of examples (should increase)
                    if 'total_size' in df.columns:
                        current_total_size = df.loc[mask, 'total_size'].iloc[0]
                        new_total_size = current_total_size + additional_count
                        df.loc[mask, 'total_size'] = new_total_size
                        print(f"  Updated level_{level} cluster {cluster_id}: total_size {current_total_size} -> {new_total_size} examples")
                    else:
                        warnings.warn(f"Level {level} cluster file missing 'total_size' column")
            else:
                warnings.warn(f"Cluster {cluster_id} not found in level_{level}_clusters.csv")

        # Save to the updated file
        df.to_csv(updated_level_file, index=False)

async def main(dataset_name, save_path="./clustering_results", sample=None):
    """Main orchestration function for extending existing clustering results."""
    
    existing_data = load_hierarchy(save_path) 
    new_examples = load_examples(dataset_name, sample)
    
    existing_eids = set(existing_data['combined_df']['example_id'].tolist())
    print(f"Found {len(existing_eids)} existing examples in combined.csv")
    
    all = []
    processed_count = 0
    skipped_count = 0
    
    # only process new examples that are not already clustered
    for example in new_examples:
        if str(example.id) in existing_eids:
            skipped_count += 1
            continue
            
        assignment = await process_single_examples(example, existing_data)
        if assignment:
            all.append(assignment)  # if 1 ex fails, the others still get added
        processed_count += 1

    print(f"Processed {processed_count} new examples, skipped {skipped_count} previously clustered examples")
    
    if all:
        extend_results(all, save_path)
        print(f"{len(all)} new assignments added")
    else:
        print("No new assignments to add")

#size vs total size? size = no of sub clusters, total size = no of examples (fixed)

# next
# process >1 example at a time
# wonder how similar (in accuracy) it is to doing entire clustering again. this could maybe give some indication of how well example fit base cluster it was put in...
# prompts are very basic, can improve
