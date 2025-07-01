import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
from typing import Dict, List, Optional, Tuple

# Page config
st.set_page_config(
    page_title="Cluster Explorer - Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load demo data
#Updated to handle a third level
# new_unthread_results/sonnet4-3layers-125/20250627_144802_level_0_clusters.csv
@st.cache_data
def load_demo_data() -> Dict[str, pd.DataFrame]:
    """Load pre-placed CSV files"""
    return {
        # updated paths for new data with 3 levels - the oNLY ting to update
        # new_unthread_results/sonnet4-3layers-290/20250627_152326_examples.csv
        # category_results/sonnet4-v2/examples.csv
        'linked_examples': pd.read_csv('category_results/raw_examples.csv'), 
        'examples': pd.read_csv('category_results/sonnet4-v3/examples.csv'),
        'level_0': pd.read_csv('category_results/sonnet4-v3/level_0_clusters.csv'),
        'level_1': pd.read_csv('category_results/sonnet4-v3/level_1_clusters.csv'),
        'level_2': pd.read_csv('category_results/sonnet4-v3/level_2_clusters.csv')
    }

# Initialize state
def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        # Auto-load demo data on first visit
        try:
            dataframes = load_demo_data()
            st.session_state.dataframes = dataframes
            st.session_state.data_loaded = True
            st.session_state.max_level = 2  # Updated to support 3 levels (0, 1, 2)
        except Exception as e:
            st.error(f"Error loading demo data: {str(e)}")
    if 'navigation_path' not in st.session_state:
        st.session_state.navigation_path = []
    if 'current_level' not in st.session_state:
        st.session_state.current_level = None
    if 'selected_cluster' not in st.session_state:
        st.session_state.selected_cluster = None
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'categories'
    if 'selected_example_id' not in st.session_state:
        st.session_state.selected_example_id = None
    if 'selected_category' not in st.session_state:
        st.session_state.selected_category = None

# Parse member_clusters from list str to int
def parse_member_clusters(member_str: str) -> List[int]:
    if pd.isna(member_str) or member_str == '':
        return []
    try:
        if 'np.int32' in member_str:
            numbers = re.findall(r'np\.int32\((\d+)\)', member_str)
            return [int(num) for num in numbers]
        else:
            return ast.literal_eval(member_str)
    except:
        return []

# Colors for top level clusters - expanded for more clusters
def get_distinct_colours():
    return [
        "#DC2626", "#059669", "#2563EB", "#7C3AED", "#EA580C", 
        "#0891B2", "#BE185D", "#65A30D", "#F59E0B", "#EF4444",
        "#10B981", "#3B82F6", "#8B5CF6", "#F97316", "#06B6D4",
        "#EC4899", "#84CC16", "#FCD34D", "#F87171", "#34D399"
    ]

# Take one color and create multiple lighter versions of it
def create_colour_variations(base_colour, num_variations):
    variations = []
    
    # Remove the # symbol and convert hex to RGB numbers
    hex_colour = base_colour.replace('#', '')
    red = int(hex_colour[0:2], 16)    
    green = int(hex_colour[2:4], 16)  
    blue = int(hex_colour[4:6], 16)   
    
    for i in range(num_variations):
        if i == 0:
            # First one is the original color
            variations.append(base_colour)
        else:
            # Make each subsequent color lighter by mixing with white (255)
            lightness = 0.3 + (i * 0.4 / (num_variations - 1))
            
            new_red = int(red + (255 - red) * lightness)
            new_green = int(green + (255 - green) * lightness)
            new_blue = int(blue + (255 - blue) * lightness)
            
            # Convert back to hex
            lighter_colour = f"#{new_red:02x}{new_green:02x}{new_blue:02x}"
            variations.append(lighter_colour)
    
    return variations

# Assign colors so each top level cluster gets a distinct one, and each child gets a related shade
def assign_colours_to_clusters(dataframes, max_level):
    colour_assignments = {}
    distinct_colours = get_distinct_colours()
    
    # Start with top level clusters, go through each
    top_level_clusters = dataframes[f'level_{max_level}'] 
    
    for i, (_, cluster_row) in enumerate(top_level_clusters.iterrows()):
        cluster_id = cluster_row['cluster_id']
        
        # Pick a distinct color 
        base_colour = distinct_colours[i % len(distinct_colours)]
        colour_assignments[(max_level, cluster_id)] = base_colour
        
        # Recursively assign colors to all descendant levels
        _assign_descendant_colors(dataframes, max_level, cluster_id, base_colour, colour_assignments)
    
    return colour_assignments

def _assign_descendant_colors(dataframes, level, cluster_id, base_colour, colour_assignments):
    """Recursively assign colors to descendant clusters"""
    if level == 0:
        return  # Base level, no children
    
    # Get the current cluster data
    current_data = dataframes[f'level_{level}']
    current_cluster = current_data[current_data['cluster_id'] == cluster_id]
    
    if len(current_cluster) == 0:
        return
    
    current_cluster = current_cluster.iloc[0]
    child_cluster_ids = parse_member_clusters(current_cluster['member_clusters'])
    
    if child_cluster_ids:
        # Create color variations for children
        child_colours = create_colour_variations(base_colour, len(child_cluster_ids))
        
        for child_index, child_id in enumerate(child_cluster_ids):
            child_colour = child_colours[child_index]
            colour_assignments[(level - 1, child_id)] = child_colour
            
            # Recursively assign colors to grandchildren
            _assign_descendant_colors(dataframes, level - 1, child_id, child_colour, colour_assignments)

# Display clusters 
def display_cluster_table(df: pd.DataFrame, level: int, colour_assignments: Dict):
    st.subheader(f"Level {level} Clusters")
    
    # Show each cluster as a row
    for _, cluster_row in df.iterrows():
        cluster_id = cluster_row['cluster_id']
        
        cluster_colour = colour_assignments.get((level, cluster_id), "#9CA3AF")  
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 2])
    
            # Colored circle
            with col1:
                st.markdown(f"""
                <div style="width: 30px; height: 30px; background-color: {cluster_colour}; 
                           border-radius: 50%; margin: 10px;"></div>
                """, unsafe_allow_html=True)
            
            # Cluster name and description in middle column
            with col2:
                st.markdown(f"**{cluster_row['name']}**")
                if 'category' in cluster_row.index and pd.notna(cluster_row['category']):
                    st.markdown(f"*Category: {cluster_row['category']}*")
                if pd.notna(cluster_row['description']):
                    description = str(cluster_row['description']).replace('<summary>', '').replace('</summary>', '').strip()
                    st.write(description)
            
            # Metrics in last column
            with col3:
                if level > 0:
                    st.metric("Sub-clusters", cluster_row['size'])
                    if 'total_size' in cluster_row and pd.notna(cluster_row['total_size']):
                        st.metric("Total conversations", cluster_row['total_size'])
                else:
                    st.metric("Conversations", cluster_row['size'])
            
            # Explore button
            category = cluster_row.get('category', 'unknown')
            if st.button(f"Explore →", key=f"cluster_{level}_{cluster_id}_{category}", use_container_width=True):
                navigate_to_cluster(level, cluster_id, cluster_row['name'])
            
            st.divider()

def navigate_to_cluster(level: int, cluster_id: int, cluster_name: str):
    """Add this cluster to our navigation path and refresh the page"""
    st.session_state.navigation_path.append({
        'level': level,
        'cluster_id': cluster_id,
        'name': cluster_name
    })
    st.session_state.selected_cluster = cluster_id
    st.session_state.current_level = level
    st.rerun()

def display_parent_stack():
    """Show all parent clusters stacked vertically"""
    if not st.session_state.navigation_path:
        return
    
    # Back button at the top
    if st.button("← Back"):
        if st.session_state.navigation_path:
            # Remove the last step
            st.session_state.navigation_path.pop()
            
            # Update our current position
            if st.session_state.navigation_path:
                last_step = st.session_state.navigation_path[-1]
                st.session_state.current_level = last_step['level']
                st.session_state.selected_cluster = last_step['cluster_id']
            else:
                # Back to the beginning
                st.session_state.current_level = None
                st.session_state.selected_cluster = None
        st.rerun()
    
    # Show each parent in the navigation path
    for step in st.session_state.navigation_path:
        st.markdown(f"**L={step['level']}: {step['name']}**")
    
    # Show current level description prominently 
    if st.session_state.navigation_path:
        current_step = st.session_state.navigation_path[-1]
        current_level = current_step['level']
        current_cluster_id = current_step['cluster_id']
        
        # Get the current cluster data to show its description
        dataframes = st.session_state.dataframes
        current_data = dataframes[f'level_{current_level}']
        current_cluster = current_data[current_data['cluster_id'] == current_cluster_id].iloc[0]
        
        if pd.notna(current_cluster['description']):
            description = str(current_cluster['description']).replace('<summary>', '').replace('</summary>', '').strip()
            st.markdown(f"{description}")
        
        st.markdown("---")

def display_examples(examples_df: pd.DataFrame, cluster_id: int):
    """Show the actual conversation examples for a base cluster"""
    # Filter to just the examples in this cluster
    cluster_examples = examples_df[examples_df['base_cluster_id'] == cluster_id]
    
    st.subheader(f"Examples in Cluster (Total: {len(cluster_examples)})")
    
    # Set up pagination if there are many examples
    examples_per_page = 10
    total_pages = (len(cluster_examples) - 1) // examples_per_page + 1
    
    if total_pages > 1:
        page = st.selectbox("Page", range(1, total_pages + 1)) - 1
    else:
        page = 0
    
    # Calculate which examples to show
    start = page * examples_per_page
    end = min(start + examples_per_page, len(cluster_examples))
    
    # Show the examples for this page
    for i, (original_index, example_row) in enumerate(cluster_examples.iloc[start:end].iterrows()):
        example_id = example_row['example_id']
        
        with st.container():
            col1, col2 = st.columns([8, 2])
            
            with col1:
                st.markdown(f"**Example {start + i + 1}:** `{example_id}`")
                st.write(f"**Summary:** {example_row['summary']}")
            
            with col2:
                if st.button("View Full", key=f"view_full_{example_id}", use_container_width=True):
                    st.session_state.selected_example_id = example_id
                    st.session_state.view_mode = 'full_example'
                    st.rerun()
            
            st.divider()

def display_full_example(linked_examples_df: pd.DataFrame, examples_df: pd.DataFrame, example_id: str):
    matching_examples = linked_examples_df[linked_examples_df['example_id'] == example_id]
    
    if len(matching_examples) == 0:
        st.error(f"Example {example_id} not found in linked examples data.")
        return
    # find the full example
    example_row = matching_examples.iloc[0]
    
    # find the summary from examples dataframe
    matching_summaries = examples_df[examples_df['example_id'] == example_id]
    summary = matching_summaries['summary'].iloc[0] if len(matching_summaries) > 0 else "Summary not found"

    # Header with back button
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("← Back to Examples", type="secondary"):
            st.session_state.view_mode = 'clusters'
            st.session_state.selected_example_id = None
            st.rerun()
    
    with col2:
        st.subheader(f"Example {example_id}")

    # Display summary above ex
    st.markdown(f"**ID:** `{example_id}`")
    st.markdown(f"**Generated Summary:** {summary}")
    
    # Display the raw inputs
    if 'inputs' in example_row.index and pd.notna(example_row['inputs']):
        st.markdown("**Raw Inputs:**")
        
        try:
            import json
            inputs_data = eval(example_row['inputs'])
            st.json(inputs_data)
        except:
            st.write(example_row['inputs'])
def display_category_overview(dataframes):
    """Show category cards as landing page"""
    st.header("Product Categories")
    
    # Get category stats
    category_stats = dataframes['level_0'].groupby('category').agg({
        'size': ['count', 'sum']  # count = # clusters, sum = total conversations
    }).round()
    
    category_stats.columns = ['clusters', 'conversations']
    category_stats = category_stats.sort_values('conversations', ascending=False)
    
    # Display as cards in a grid
    cols = st.columns(2)  # 2 columns
    
    for i, (category, stats) in enumerate(category_stats.iterrows()):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"### {category}")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Clusters", int(stats['clusters']))
                with col2:
                    st.metric("Conversations", int(stats['conversations']))
                
                if st.button(f"Explore {category} →", key=f"category_{i}", use_container_width=True):
                    st.session_state.selected_category = category
                    st.session_state.view_mode = 'category_clusters'
                    st.rerun()
                
                st.markdown("---")

def display_category_clusters(dataframes, category):
    """Show clusters for a specific category"""
    # Back button
    if st.button("← Back to Categories"):
        st.session_state.selected_category = None
        st.session_state.view_mode = 'categories'
        st.rerun()
    
    st.subheader(f"{category} - Clusters")
    
    # Filter to category
    category_data = dataframes['level_0'][dataframes['level_0']['category'] == category]
    category_data = category_data.sort_values('size', ascending=False)
    
    display_cluster_table(category_data, 0, st.session_state.colour_assignments)


def main():
    st.title("OpenCLIO Insights on Customer Support Data")
    st.markdown("See 2990 customer support conversations organized into hierarchical clusters. You can navigate from 5 high-level topic areas down through 15 mid-level clusters, then to 125 specific clusters, and finally to conversation summaries and full Chat LangChain threads.")
    
    init_session_state()

    dataframes = st.session_state.dataframes
    
    # Sort each level by size (descending)
    for level_key in ['level_0', 'level_1', 'level_2']:
        if level_key in dataframes:
            size_col = 'total_size' if level_key != 'level_0' else 'size'
            if size_col in dataframes[level_key].columns:
                dataframes[level_key] = dataframes[level_key].sort_values(size_col, ascending=False)
    
    # Sidebar with dataset info
    with st.sidebar:
        st.header("Customer Support Insights Overview")
        st.markdown("### Dataset Summary")
        
        # calc stats
        total_conversations = dataframes['level_0']['size'].sum()
        level_0_count = len(dataframes['level_0'])
        level_1_count = len(dataframes.get('level_1', []))
        level_2_count = len(dataframes.get('level_2', []))

        # and display them
        st.metric("Total Support Conversations", f"{total_conversations:,}")
        
        # Make hierarchy clearer
        st.divider()
        st.markdown("### How to Navigate")
        st.markdown("**Categories** → Product areas (Admin, LangSmith, etc.)")
        st.markdown("**Clusters** → Common issue types within each product")
        st.markdown("**Conversations** → Individual support tickets")
        st.divider()

        st.markdown("### Support Volume by Product")
        categories = dataframes['level_0'].groupby('category')['size'].sum().sort_values(ascending=False)
    
        for category, conv_count in categories.items():
            percentage = (conv_count / total_conversations) * 100
            st.markdown(f"**{category}**: {conv_count:,} conversations ({percentage:.1f}%)")
    
        # Add insights
        st.divider()
        st.markdown("### Quick Insights") #TODO
        st.info(f"[can put anything else we'd want pinned/to see instantly here]")
        st.success(f"[like this]")
    
        # Show current location if navigating
        if st.session_state.get('selected_category'):
            st.markdown("---")
            st.markdown(f"**Currently viewing:** {st.session_state.selected_category}")
    
    # Show the navigation breadcrumb
    display_parent_stack()
    
    # dataframes = st.session_state.dataframes
    
    # Create the color scheme once and store it
    if 'colour_assignments' not in st.session_state:
        st.session_state.colour_assignments = assign_colours_to_clusters(dataframes, st.session_state.max_level)
    
    colour_assignments = st.session_state.colour_assignments
    
    # Decide what to show based on where we are in the navigation
    if st.session_state.view_mode == 'full_example':
        # Show full example details
        display_full_example(dataframes['linked_examples'], dataframes['examples'], st.session_state.selected_example_id)
    
    elif st.session_state.view_mode == 'categories':
        display_category_overview(dataframes)
    elif st.session_state.view_mode == 'category_clusters':
        display_category_clusters(dataframes, st.session_state.selected_category)
    elif not st.session_state.navigation_path:
        # This is now the fallback
        base_level_data = dataframes['level_0']
        display_cluster_table(base_level_data, 0, colour_assignments)
    else:
        # We've navigated, now figure out what to show
        current_step = st.session_state.navigation_path[-1]
        current_level = current_step['level']
        current_cluster_id = current_step['cluster_id']
        
        if current_level == 0:
            # We're at the bottom level - show individual examples
            display_examples(dataframes['examples'], current_cluster_id)
        else:
            # We're at a middle level, show its child clusters
            current_data = dataframes[f'level_{current_level}']
            current_cluster = current_data[current_data['cluster_id'] == current_cluster_id].iloc[0]

            # Find the children of this cluster
            child_cluster_ids = parse_member_clusters(current_cluster['member_clusters'])
            
            if child_cluster_ids:
                # Get the data for the child level
                child_level = current_level - 1
                child_data = dataframes[f'level_{child_level}']
                children = child_data[child_data['cluster_id'].isin(child_cluster_ids)]
                
                display_cluster_table(children, child_level, colour_assignments)
            else:
                st.warning("No child clusters found for this selection")

    

if __name__ == "__main__":
    main()
