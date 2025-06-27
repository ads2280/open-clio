import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langsmith import Client
import umap
import os
from matplotlib.patches import Patch

def load_embeddings_and_clusters(experiment_name=None, dataset_name="open-clio-data-test", regenerate_embeddings=True):
    """
    Load embeddings and cluster assignments from your experiment.
    If embeddings file doesn't exist, regenerate them from the summaries.
    """
    embeddings = None
    
    # Try to load existing embeddings if experiment_name is provided
    if experiment_name:
        embeddings_file = f"experiment_results/{experiment_name}/embeddings.npy"
        if os.path.exists(embeddings_file):
            embeddings = np.load(embeddings_file)
            print(f"Loaded existing embeddings with shape: {embeddings.shape}")
        else:
            print(f"Embeddings file not found: {embeddings_file}")
    
    # Load cluster assignments from LangSmith
    client = Client()
    examples = list(client.list_examples(dataset_name=dataset_name))
    
    cluster_labels = []
    cluster_names = []
    summaries = []
    
    for example in examples:
        if 'cluster_id' in example.outputs:
            cluster_labels.append(example.outputs['cluster_id'])
            cluster_names.append(example.outputs.get('cluster_name', f"Cluster {example.outputs['cluster_id']}"))
            summaries.append(example.outputs['request'])
        else:
            print("Warning: Some examples don't have cluster assignments")
    
    cluster_labels = np.array(cluster_labels)
    
    print(f"Loaded {len(cluster_labels)} cluster assignments")
    print(f"Number of unique clusters: {len(np.unique(cluster_labels))}")
    
    # Generate embeddings if we don't have them
    if embeddings is None and regenerate_embeddings:
        print("Generating embeddings from summaries...")
        import openai
        
        # Use the same embedding model as your clustering script
        embedding_model = "text-embedding-3-small"
        
        # Generate embeddings using OpenAI
        response = openai.embeddings.create(
            model=embedding_model,
            input=summaries
        )
        
        # Extract embedding vectors
        embeddings = np.array([item.embedding for item in response.data])
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        # Save embeddings for future use
        if experiment_name:
            embeddings_file = f"experiment_results/{experiment_name}/embeddings.npy"
            os.makedirs(os.path.dirname(embeddings_file), exist_ok=True)
            np.save(embeddings_file, embeddings)
            print(f"Saved embeddings to: {embeddings_file}")
        else:
            # Save with timestamp if no experiment name
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            embeddings_file = f"embeddings_{timestamp}.npy"
            np.save(embeddings_file, embeddings)
            print(f"Saved embeddings to: {embeddings_file}")
    
    if embeddings is None:
        raise ValueError("Could not load or generate embeddings")
    
    return embeddings, cluster_labels, cluster_names, summaries

def create_umap_projection(embeddings, n_neighbors=15, min_dist=0, metric='cosine', random_state=42):
    """
    Create UMAP projection using the same parameters as the Clio paper
    From Appendix G.6: n_neighbors = 15, min_dist = 0, cosine metric
    """
    print("Creating UMAP projection with Clio paper parameters...")
    print(f"Parameters: n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2
    )
    
    embedding_2d = reducer.fit_transform(embeddings)
    print(f"UMAP projection complete. 2D embedding shape: {embedding_2d.shape}")
    
    return embedding_2d, reducer

def plot_clusters(embedding_2d, cluster_labels, cluster_names=None, figsize=(20, 12), 
                 title="UMAP Projection of Embeddings Colored by Base Clusters", 
                 save_path=None, show_legend=True):
    """
    Create scatter plot of UMAP projection colored by cluster labels with comprehensive legend
    """
    # Create color palette for clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    # Use a color palette that works well for many categories
    if n_clusters <= 10:
        colors = sns.color_palette("tab10", n_clusters)
    elif n_clusters <= 20:
        colors = sns.color_palette("tab20", n_clusters)
    else:
        colors = sns.color_palette("husl", n_clusters)
    
    # Create subplot layout: main plot + legend
    fig = plt.figure(figsize=figsize)
    
    # Main plot takes up most of the space
    ax_main = plt.subplot2grid((1, 4), (0, 0), colspan=3)
    
    # Plot each cluster
    legend_info = []
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        cluster_points = embedding_2d[mask]
        
        # Get cluster name if available
        if cluster_names is not None:
            cluster_name_list = [cluster_names[j] for j in range(len(cluster_labels)) if cluster_labels[j] == cluster_id]
            if cluster_name_list:
                cluster_name = cluster_name_list[0]
                label = f"{cluster_id}: {cluster_name[:40]}..." if len(cluster_name) > 40 else f"{cluster_id}: {cluster_name}"
            else:
                label = f"Cluster {cluster_id}"
                cluster_name = "No name available"
        else:
            label = f"Cluster {cluster_id}"
            cluster_name = "No name available"
        
        ax_main.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[i]], alpha=0.7, s=25)
        
        # Store info for legend
        legend_info.append({
            'id': cluster_id,
            'name': cluster_name,
            'color': colors[i],
            'count': len(cluster_points),
            'percentage': len(cluster_points)/len(cluster_labels)*100
        })
    
    ax_main.set_title(title, fontsize=16, fontweight='bold')
    ax_main.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax_main.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax_main.grid(True, alpha=0.3)
    
    # Create legend subplot
    if show_legend:
        ax_legend = plt.subplot2grid((1, 4), (0, 3))
        ax_legend.axis('off')
        
        # Sort clusters by ID for organized legend
        legend_info.sort(key=lambda x: x['id'])
        
        # Create legend content
        legend_text = "Cluster Legend:\n\n"
        y_positions = []
        colors_for_legend = []
        
        for i, info in enumerate(legend_info):
            # Truncate very long names for legend
            display_name = info['name'][:35] + "..." if len(info['name']) > 35 else info['name']
            
            legend_text += f"{info['id']}: {display_name}\n"
            legend_text += f"    ({info['count']} points, {info['percentage']:.1f}%)\n\n"
            
            y_positions.append(0.95 - i * 0.08)  # Spacing for legend items
            colors_for_legend.append(info['color'])
        
        # Add text to legend
        ax_legend.text(0.05, 0.95, legend_text, fontsize=8, verticalalignment='top',
                      fontfamily='monospace', transform=ax_legend.transAxes)
        
        # Add color squares
        for i, (y_pos, color) in enumerate(zip(y_positions, colors_for_legend)):
            if y_pos > 0:  # Only draw if within bounds
                ax_legend.add_patch(plt.Rectangle((0.01, y_pos-0.01), 0.03, 0.02, 
                                                facecolor=color, transform=ax_legend.transAxes))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print cluster statistics
    print(f"\nCluster Statistics:")
    print(f"Total points: {len(cluster_labels)}")
    print(f"Number of clusters: {n_clusters}")
    
    # Print detailed cluster info
    for info in sorted(legend_info, key=lambda x: x['id']):
        print(f"Cluster {info['id']}: {info['count']} points ({info['percentage']:.1f}%) - {info['name'][:50]}...")

def create_separate_legend(cluster_labels, cluster_names, colors, save_path=None):
    """
    Create a separate comprehensive legend figure for cases with many clusters
    """
    unique_clusters = np.unique(cluster_labels)
    
    # Calculate figure height based on number of clusters
    fig_height = max(8, len(unique_clusters) * 0.3)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.axis('off')
    
    # Prepare legend info
    legend_info = []
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_labels == cluster_id
        count = np.sum(mask)
        percentage = count/len(cluster_labels)*100
        
        if cluster_names is not None:
            cluster_name_list = [cluster_names[j] for j in range(len(cluster_labels)) if cluster_labels[j] == cluster_id]
            cluster_name = cluster_name_list[0] if cluster_name_list else "No name available"
        else:
            cluster_name = "No name available"
        
        legend_info.append({
            'id': cluster_id,
            'name': cluster_name,
            'color': colors[i],
            'count': count,
            'percentage': percentage
        })
    
    # Sort by cluster ID
    legend_info.sort(key=lambda x: x['id'])
    
    # Create legend with color boxes and text
    y_start = 0.95
    line_height = 0.8 / len(legend_info)  # Distribute evenly
    
    for i, info in enumerate(legend_info):
        y_pos = y_start - i * line_height
        
        # Color box
        ax.add_patch(plt.Rectangle((0.02, y_pos-0.015), 0.04, 0.03, 
                                  facecolor=info['color'], transform=ax.transAxes,
                                  edgecolor='black', linewidth=0.5))
        
        # Cluster text
        cluster_text = f"Cluster {info['id']}: {info['name']}"
        stats_text = f"({info['count']} points, {info['percentage']:.1f}%)"
        
        ax.text(0.08, y_pos, cluster_text, fontsize=10, fontweight='bold',
               verticalalignment='center', transform=ax.transAxes)
        ax.text(0.08, y_pos-0.02, stats_text, fontsize=8, color='gray',
               verticalalignment='center', transform=ax.transAxes)
    
    plt.title("Cluster Legend", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Legend saved to: {save_path}")
    
    plt.show()
    return fig
def create_interactive_plot(embedding_2d, cluster_labels, summaries, cluster_names=None):
    """
    Create an interactive plotly visualization (optional - requires plotly)
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd
        
        # Create DataFrame for plotly
        df = pd.DataFrame({
            'x': embedding_2d[:, 0],
            'y': embedding_2d[:, 1],
            'cluster': cluster_labels,
            'summary': summaries,
            'cluster_name': cluster_names if cluster_names else [f"Cluster {c}" for c in cluster_labels]
        })
        
        # Create hover text
        df['hover_text'] = df.apply(lambda row: f"Cluster {row['cluster']}<br>{row['cluster_name']}<br><br>{row['summary'][:100]}...", axis=1)
        
        fig = px.scatter(df, x='x', y='y', color='cluster', 
                        hover_name='hover_text',
                        title='Interactive UMAP Projection of Embeddings',
                        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'})
        
        fig.update_traces(marker=dict(size=5, opacity=0.7))
        fig.update_layout(width=1200, height=800)
        
        return fig
    
    except ImportError:
        print("Plotly not available. Skipping interactive plot.")
        return None

def main(experiment_name=None, dataset_name="open-clio-data-test"):
    """
    Main function to create UMAP visualization
    """
    # Load data (will generate embeddings if needed)
    embeddings, cluster_labels, cluster_names, summaries = load_embeddings_and_clusters(
        experiment_name, dataset_name
    )
    
    # Create UMAP projection with Clio paper parameters
    embedding_2d, reducer = create_umap_projection(embeddings)
    
    # Determine output directory
    if experiment_name:
        output_dir = f"experiment_results/{experiment_name}"
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"umap_results_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create static plot with legend
    save_path = f"{output_dir}/umap_clusters.png"
    
    plot_clusters(embedding_2d, cluster_labels, cluster_names, 
                 save_path=save_path)
    
    # Create separate detailed legend if many clusters
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    
    if n_clusters > 15:  # Create separate legend for better readability
        if n_clusters <= 10:
            colors = sns.color_palette("tab10", n_clusters)
        elif n_clusters <= 20:
            colors = sns.color_palette("tab20", n_clusters)
        else:
            colors = sns.color_palette("husl", n_clusters)
            
        legend_path = f"{output_dir}/cluster_legend.png"
        create_separate_legend(cluster_labels, cluster_names, colors, legend_path)
        print(f"Detailed legend saved to: {legend_path}")
    
    # Create interactive plot if possible
    interactive_fig = create_interactive_plot(embedding_2d, cluster_labels, summaries, cluster_names)
    if interactive_fig:
        interactive_path = f"{output_dir}/umap_clusters_interactive.html"
        interactive_fig.write_html(interactive_path)
        print(f"Interactive plot saved to: {interactive_path}")
    
    # Save the 2D embeddings for future use
    embedding_2d_path = f"{output_dir}/umap_2d_embeddings.npy"
    np.save(embedding_2d_path, embedding_2d)
    print(f"2D embeddings saved to: {embedding_2d_path}")
    
    return embedding_2d, reducer

if __name__ == "__main__":
    # You can now run this in several ways:
    
    # Option 1: If you have an experiment name and saved embeddings
    # experiment_name = "your_experiment_name_here"
    # embedding_2d, reducer = main(experiment_name)
    
    # Option 2: Generate embeddings from LangSmith data (recommended for your case)
    experiment_name = None  # Will generate a timestamped directory
    embedding_2d, reducer = main(experiment_name)
    
    # Option 3: Provide a custom experiment name for saving
    # experiment_name = "langchain_clustering_analysis"
    # embedding_2d, reducer = main(experiment_name)
    
    print("UMAP visualization complete!")