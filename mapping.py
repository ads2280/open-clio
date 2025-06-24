import umap
import pandas as pd
import numpy as np
import sentence_transformers 

def examples_to_base_cluster(embeddings, cluster_labels, example_ids, summaries):
    projection = umap.UMAP(
        n_neighbors=15,
        min_dist=0,
        # n_components= 2, 
        metric='cosine'
    )

    coords_2d = projection.fit_transform(embeddings)
    
    # Save as DataFrame
    df_projection = pd.DataFrame({
        'example_id': example_ids,
        'x': coords_2d[:, 0],
        'y': coords_2d[:, 1],
        'cluster_id': cluster_labels,
        'summary': summaries
    })
    
    return df_projection

def base_cluster_to_top(hierarchy):
    base_clusters = hierarchy['level_0']
    cluster_descriptions = []
    cluster_ids = []

    for cluster_id, info in base_clusters.items():
        text = f"{info['name']}: {info['description']}"
        cluster_descriptions.append(text)
        cluster_ids.append(cluster_id)

    # embed cluster descriptions 
    embedder = sentence_transformers.SentenceTransformer("all-mpnet-base-v2")
    cluster_embeddings = embedder.encode(cluster_descriptions)
    
    # Project base clusters to 2D
    project_again = umap.UMAP(
        n_neighbors=10,  # 15 in clio, but artifically smaller
        min_dist=0,
        metric='cosine'
    )
    
    cluster_coords_2d = project_again.fit_transform(cluster_embeddings)
    
    # Create mapping from base cluster to higher-level cluster
    base_to_higher = {}
    if hierarchy['max_level'] > 0:
        level_1_clusters = hierarchy['level_1']
        for hl_id, hl_info in level_1_clusters.items():
            for base_cluster_id in hl_info['member_clusters']:
                base_to_higher[base_cluster_id] = {
                    'higher_level_id': hl_id,
                    'higher_level_name': hl_info['name']
                }
    
    # Create DataFrame
    df_projection_helper = []
    for i, cluster_id in enumerate(cluster_ids):
        base_info = base_clusters[cluster_id]
        higher_info = base_to_higher.get(cluster_id, {'higher_level_id': 0, 'higher_level_name': 'Ungrouped'})
        
        df_projection_helper.append({
            'base_cluster_id': cluster_id,
            'base_cluster_name': base_info['name'],
            'base_cluster_size': base_info['size'],
            'higher_level_id': higher_info['higher_level_id'],
            'higher_level_name': higher_info['higher_level_name'],
            'x': cluster_coords_2d[i, 0],
            'y': cluster_coords_2d[i, 1]
        })
    
    df_projection_again = pd.DataFrame(df_projection_helper)
    return df_projection_again
