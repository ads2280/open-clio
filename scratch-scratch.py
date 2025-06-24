
print("SAVING CLUSTERING RESULTS")

# Save everything to DataFrames and CSV files
dataframes = save_results(
    hierarchy=hierarchy,
    example_ids=example_ids, 
    cluster_labels=cluster_labels,
    summaries=summaries,
    output_dir="clustering_results"
)

print("\n" + "="*60)
print("CLIO HIERARCHICAL CLUSTERING COMPLETE!")
print("="*60)

# Display quick summary of what was saved
print(f"\nGenerated {len(dataframes)} DataFrames:")
for name, df in dataframes.items():
    print(f"- {name}: {len(df)} rows, {len(df.columns)} columns")

print(f"\nDataFrames are available as:")
for name in dataframes.keys():
    print(f"- dataframes['{name}']")

print(f"\nAll results saved to 'clustering_results/' directory")

# Optional: Display a sample of the most useful DataFrame
print(f"\nSample of full_hierarchy DataFrame (first 3 rows):")
print(dataframes['full_hierarchy'].head(3).to_string())

# Optional: You can also access individual DataFrames like this:
# df_examples = dataframes['examples']
# df_clusters = dataframes['cluster_details'] 
# df_hierarchy = dataframes['hierarchy_mapping']
# df_summary = dataframes['level_summary']
# df_full = dataframes['full_hierarchy']


            # for reference:
            #cluster_info[cluster_id] = {
            #'name': name,
            #'description': summary,
            #'size': len(cluster_summaries),
            #'summaries': cluster_summaries[:10] # just first 10 for inspection
            #}

            # criteria




            #where is cluster_name var
            # make a doc with the properties of clusters - id, info is a dict wihch has name and description

        


        


        # next:
        # criteria and prompts
        # clusters_per_neighborhood and desired_names are the same - fix
        #scratchpads are so cute - for promptimizer. helpful to learn about prompting. wonder if it has the same effect as agent
        # for later- prompts in prompt hub? or something other way to not have them take up 99% of lines
        # how does human/assistant - like feeding both into messages work
        

    # skipping privacy auditor for now
        
        


    

    

    

    # this process continues until reaching ktop
        


        # printing stuff

print("Hierarchical clustering complete!!")

# Print summary statistics
max_level = hierarchy['max_level']
print(f"\nHierarchy Summary:")
print(f"Total levels created: {max_level + 1} (0 to {max_level})")

for level in range(max_level + 1):
    level_clusters = hierarchy[f'level_{level}']
    print(f"Level {level}: {len(level_clusters)} clusters")
    
    if level == 0:
        total_conversations = sum(info['size'] for info in level_clusters.values())
        print(f"  Total conversations: {total_conversations}")
    else:
        total_size = sum(info['total_size'] for info in level_clusters.values())
        print(f"  Total conversations: {total_size}")














# initial save as dataframes method
# create output dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dataframes = {}
    
    # 1) Base data i.e. original examples with cluster assignments
    base_data = []
    for i, (example_id, cluster_id) in enumerate(zip(example_ids, cluster_labels)):
        base_data.append({
            'example_id': example_id,
            'summary': summaries[i],
            'base_cluster_id': int(cluster_id),
            'base_cluster_name': hierarchy['level_0'][cluster_id]['name'],
            'base_cluster_description': hierarchy['level_0'][cluster_id]['description']
        })
    
    df_examples = pd.DataFrame(base_data)
    dataframes['examples'] = df_examples
    
    # 2) Cluster details, i.e. information about each cluster at each level
    cluster_details = []
    max_level = hierarchy['max_level']
    
    for level in range(max_level + 1):
        level_clusters = hierarchy[f'level_{level}']
        
        for cluster_id, info in level_clusters.items():
            cluster_details.append({
                'level': level,
                'cluster_id': cluster_id,
                'name': info['name'],
                'description': info['description'],
                'size': info['size'],
                'total_size': info.get('total_size', info['size']),
                'member_clusters': str(info.get('member_clusters', [])),  # Convert list to string for CSV
                'sample_summaries': str(info.get('summaries', [])[:3])  # First 3 summaries as string
            })
    
    df_clusters = pd.DataFrame(cluster_details)
    dataframes['cluster_details'] = df_clusters
    
    # 3) Hierarchy mapping that shows parent-child relationships
    hierarchy_mapping = []
    
    for level in range(1, max_level + 1):
        level_clusters = hierarchy[f'level_{level}']
        
        for parent_id, parent_info in level_clusters.items():
            for child_id in parent_info.get('member_clusters', []):
                hierarchy_mapping.append({
                    'parent_level': level,
                    'parent_cluster_id': parent_id,
                    'parent_name': parent_info['name'],
                    'child_level': level - 1,
                    'child_cluster_id': child_id,
                    'child_name': hierarchy[f'level_{level-1}'][child_id]['name']
                })
    
    df_hierarchy = pd.DataFrame(hierarchy_mapping)
    dataframes['hierarchy_mapping'] = df_hierarchy
    
    # 4) Level summary for high-level statistics
    level_summary = []
    for level in range(max_level + 1):
        level_clusters = hierarchy[f'level_{level}']
        total_conversations = sum(info.get('total_size', info['size']) for info in level_clusters.values())
        
        level_summary.append({
            'level': level,
            'num_clusters': len(level_clusters),
            'total_conversations': total_conversations,
            'avg_cluster_size': total_conversations / len(level_clusters) if level_clusters else 0
        })
    
    df_summary = pd.DataFrame(level_summary)
    dataframes['level_summary'] = df_summary
    
    # 5) full flattened view, each example with all its hierarchical assignments
    full_data = []
    
    # Create mapping from base cluster to all parent clusters
    base_to_parents = {}
    for example_id, base_cluster_id in zip(example_ids, cluster_labels):
        parents = {0: base_cluster_id}  # Level 0 is the base cluster
        
        # Trace up the hierarchy
        current_cluster = base_cluster_id
        for level in range(1, max_level + 1):
            level_clusters = hierarchy[f'level_{level}']
            for parent_id, parent_info in level_clusters.items():
                if current_cluster in parent_info.get('member_clusters', []):
                    parents[level] = parent_id
                    current_cluster = parent_id
                    break
        
        base_to_parents[base_cluster_id] = parents
    
    # Build full flattened data
    for i, (example_id, base_cluster_id) in enumerate(zip(example_ids, cluster_labels)):
        row = {
            'example_id': example_id,
            'summary': summaries[i],
        }
        
        # Add cluster info for each level
        parents = base_to_parents[base_cluster_id]
        for level in range(max_level + 1):
            if level in parents:
                cluster_id = parents[level]
                cluster_info = hierarchy[f'level_{level}'][cluster_id]
                row[f'level_{level}_cluster_id'] = cluster_id
                row[f'level_{level}_cluster_name'] = cluster_info['name']
                row[f'level_{level}_cluster_description'] = cluster_info['description']
            else:
                row[f'level_{level}_cluster_id'] = None
                row[f'level_{level}_cluster_name'] = None
                row[f'level_{level}_cluster_description'] = None
        
        full_data.append(row)
    
    df_full = pd.DataFrame(full_data)
    dataframes['full_hierarchy'] = df_full
    
    # Save all DataFrames to CSV files
    saved_files = []
    for name, df in dataframes.items():
        filename = f"{output_dir}/{timestamp}_{name}.csv"
        df.to_csv(filename, index=False)
        saved_files.append(filename)
        print(f"Saved {name}: {filename} ({len(df)} rows)")
    
    # Save a summary report
    report_filename = f"{output_dir}/{timestamp}_clustering_report.txt"
    with open(report_filename, 'w') as f:
        f.write("CLIO HIERARCHICAL CLUSTERING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Total examples processed: {len(example_ids)}\n")
        f.write(f"Hierarchy levels: {max_level + 1} (0 to {max_level})\n\n")
        
        f.write("LEVEL SUMMARY:\n")
        f.write("-" * 20 + "\n")
        for _, row in df_summary.iterrows():
            f.write(f"Level {int(row['level'])}: {int(row['num_clusters'])} clusters, "
                   f"{int(row['total_conversations'])} total items\n")
        
        f.write("\nFILES GENERATED:\n")
        f.write("-" * 20 + "\n")
        for filename in saved_files:
            f.write(f"- {filename}\n")
        
        f.write(f"\nFILE DESCRIPTIONS:\n")
        f.write("-" * 20 + "\n")
        f.write("- examples.csv: Original data with base cluster assignments\n")
        f.write("- cluster_details.csv: Information about each cluster at each level\n")
        f.write("- hierarchy_mapping.csv: Parent-child relationships between clusters\n")
        f.write("- level_summary.csv: High-level statistics by level\n")
        f.write("- full_hierarchy.csv: Complete flattened view with all hierarchical assignments\n")
    
    print(f"\nSaved clustering report: {report_filename}")
    
    return dataframes


    




















# cluster_names = []
# already have example_ids list
#for example in examples:
#    cluster_names.append(example.outputs["cluster_name"]) #i.e. base cluster

#print(f"loaded {len(cluster_names)} cluster names)")

#embeddings_2 = embedder.encode(cluster_name)
#print(print(f"first round embeddings generated, shape: {embeddings_2.shape}"))

#k = math.sqrt(len(summaries))



# UMAP experimentation
df_2d = examples_to_base_cluster(embeddings, cluster_labels, example_ids, summaries)
print('\n\n\n\numap experiments')

print(f'\nexamples to base cluster')

# save to CSV
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df_2d.to_csv(f"clustering_results/{timestamp}_2d_coordinates.csv", index=False)
print(f"Saved 2D coordinates: {len(df_2d)} points")

print(f'\nnow doing base cluster to top')
df_2d_higher = base_cluster_to_top(hierarchy)

# save to CSV
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
df_2d_higher.to_csv(f"clustering_results/{timestamp}_hierarchical_visualization.csv", index=False)
print(f"Saved hierarchical visualization: {len(df_2d_higher)} base clusters")
print("Columns: base_cluster_id, base_cluster_name, higher_level_id, higher_level_name, x, y")