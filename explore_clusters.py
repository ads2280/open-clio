import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
from typing import Dict, List, Optional, Tuple

# Page configuration
st.set_page_config(
    page_title="Cluster Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    """Initialize session state variables for navigation"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'navigation_path' not in st.session_state:
        st.session_state.navigation_path = []
    if 'current_level' not in st.session_state:
        st.session_state.current_level = None
    if 'selected_cluster' not in st.session_state:
        st.session_state.selected_cluster = None
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = 'clusters'  # 'clusters' or 'examples'

@st.cache_data
def load_csv_files(files_dict: Dict[str, any]) -> Dict[str, pd.DataFrame]:
    """Load and cache CSV files"""
    dataframes = {}
    
    for file_type, uploaded_file in files_dict.items():
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            dataframes[file_type] = df
    
    return dataframes

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

def get_distinct_colors():
    """Get 8 distinct colors for the top-level clusters"""
    return [
        "#DC2626",  # Red
        "#059669",  # Green  
        "#2563EB",  # Blue
        "#7C3AED",  # Purple
        "#EA580C",  # Orange
        "#0891B2",  # Cyan
        "#BE185D",  # Pink
        "#65A30D",  # Lime
    ]

def create_color_variations(base_color, num_variations):
    """
    Take one color and create multiple lighter versions of it
    For example: red -> red, light red, lighter red, very light red
    """
    variations = []
    
    # Remove the # symbol and convert hex to RGB numbers
    hex_color = base_color.replace('#', '')
    red = int(hex_color[0:2], 16)    # First 2 characters
    green = int(hex_color[2:4], 16)  # Next 2 characters  
    blue = int(hex_color[4:6], 16)   # Last 2 characters
    
    for i in range(num_variations):
        if i == 0:
            # First one is the original color
            variations.append(base_color)
        else:
            # Make each subsequent color lighter by mixing with white (255)
            lightness = 0.3 + (i * 0.4 / (num_variations - 1))
            
            new_red = int(red + (255 - red) * lightness)
            new_green = int(green + (255 - green) * lightness)
            new_blue = int(blue + (255 - blue) * lightness)
            
            # Convert back to hex
            lighter_color = f"#{new_red:02x}{new_green:02x}{new_blue:02x}"
            variations.append(lighter_color)
    
    return variations

def assign_colors_to_clusters(dataframes, max_level):
    """
    Assign colors so that:
    - Each top-level cluster gets a distinct color (red, green, blue, etc.)
    - Each child cluster gets a lighter shade of its parent's color
    """
    color_assignments = {}
    distinct_colors = get_distinct_colors()
    
    # Get the top-level clusters
    top_level_clusters = dataframes[f'level_{max_level}']
    
    # Go through each top-level cluster
    for i, (_, cluster_row) in enumerate(top_level_clusters.iterrows()):
        cluster_id = cluster_row['cluster_id']
        
        # Pick a distinct color for this top-level cluster
        base_color = distinct_colors[i % len(distinct_colors)]
        color_assignments[(max_level, cluster_id)] = base_color
        
        # Now find all the children of this cluster and give them lighter shades
        if max_level > 0:  # Only if there are child levels
            child_ids = parse_member_clusters(cluster_row['member_clusters'])
            
            if child_ids:
                # Create lighter variations of the parent color
                child_colors = create_color_variations(base_color, len(child_ids))
                
                # Assign each child a shade of the parent color
                for child_index, child_id in enumerate(child_ids):
                    color_assignments[(max_level - 1, child_id)] = child_colors[child_index]
    
    return color_assignments

def display_cluster_table(df: pd.DataFrame, level: int, color_assignments: Dict):
    """Display clusters with their hierarchical colors"""
    st.subheader(f"Level {level} Clusters")
    
    # Show each cluster as a row
    for _, cluster_row in df.iterrows():
        cluster_id = cluster_row['cluster_id']
        
        # Look up the color for this cluster
        cluster_color = color_assignments.get((level, cluster_id), "#9CA3AF")
        
        # Create the visual layout
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 2])
            
            # Color circle in first column
            with col1:
                st.markdown(f"""
                <div style="width: 30px; height: 30px; background-color: {cluster_color}; 
                           border-radius: 50%; margin: 10px;"></div>
                """, unsafe_allow_html=True)
            
            # Cluster name and description in middle column
            with col2:
                st.markdown(f"**{cluster_row['name']}**")
                if pd.notna(cluster_row['description']):
                    # Clean up the description text
                    description = str(cluster_row['description']).replace('<summary>', '').replace('</summary>', '').strip()
                    shortened = description[:200] + "..." if len(description) > 200 else description
                    st.write(shortened)
            
            # Metrics in last column
            with col3:
                st.metric("Size", cluster_row['size'])
                if 'total_size' in cluster_row and pd.notna(cluster_row['total_size']) and cluster_row['total_size'] != cluster_row['size']:
                    st.metric("Total", cluster_row['total_size'])
            
            # Explore button
            if st.button(f"Explore →", key=f"cluster_{level}_{cluster_id}", use_container_width=True):
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

def display_breadcrumb():
    """Show where we are in the navigation"""
    if not st.session_state.navigation_path:
        return
    
    # Build the breadcrumb trail
    trail = ["🏠 Home"]
    for step in st.session_state.navigation_path:
        short_name = step['name'][:30] + "..." if len(step['name']) > 30 else step['name']
        trail.append(f"Level {step['level']}: {short_name}")
    
    breadcrumb = " → ".join(trail)
    st.markdown(f"**Navigation:** {breadcrumb}")
    
    # Back button
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
    for i, (_, example_row) in enumerate(cluster_examples.iloc[start:end].iterrows()):
        with st.expander(f"Example {start + i + 1}: {example_row['example_id'][:8]}..."):
            st.write("**Summary:**")
            st.write(example_row['summary'])
            st.write(f"**Example ID:** `{example_row['example_id']}`")

def main():
    st.title("🔍 Hierarchical Cluster Explorer")
    st.markdown("Navigate through your cluster hierarchy and explore conversation examples")
    
    init_session_state()
    
    # Sidebar for uploading files
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # Required files
        examples_file = st.file_uploader("Examples CSV", type=['csv'], key="examples")
        level_0_file = st.file_uploader("Level 0 Clusters CSV", type=['csv'], key="level0")
        level_1_file = st.file_uploader("Level 1 Clusters CSV", type=['csv'], key="level1")
        
        # Optional additional levels
        st.markdown("**Optional Higher Levels:**")
        level_2_file = st.file_uploader("Level 2 Clusters CSV", type=['csv'], key="level2")
        level_3_file = st.file_uploader("Level 3 Clusters CSV", type=['csv'], key="level3")
        
        # Load button
        if st.button("Load Data"):
            files = {
                'examples': examples_file,
                'level_0': level_0_file,
                'level_1': level_1_file,
                'level_2': level_2_file,
                'level_3': level_3_file
            }
            
            # Remove empty file slots
            files = {name: file for name, file in files.items() if file is not None}
            
            # Check we have the minimum required files
            if 'examples' in files and 'level_0' in files:
                try:
                    # Load all the CSV files
                    dataframes = load_csv_files(files)
                    st.session_state.dataframes = dataframes
                    st.session_state.data_loaded = True
                    
                    # Figure out what the highest level is
                    level_numbers = [int(name.split('_')[1]) for name in dataframes.keys() if 'level_' in name]
                    st.session_state.max_level = max(level_numbers)
                    
                    st.success("Data loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            else:
                st.warning("Please upload at least the Examples and Level 0 files")
    
    # Main area
    if not st.session_state.data_loaded:
        st.info("👆 Please upload your CSV files in the sidebar to get started")
        return
    
    # Show the navigation breadcrumb
    display_breadcrumb()
    
    dataframes = st.session_state.dataframes
    
    # Create the color scheme once and store it
    if 'color_assignments' not in st.session_state:
        st.session_state.color_assignments = assign_colors_to_clusters(dataframes, st.session_state.max_level)
    
    color_assignments = st.session_state.color_assignments
    
    # Decide what to show based on where we are in the navigation
    if not st.session_state.navigation_path:
        # We're at the top - show the highest level clusters
        max_level = st.session_state.max_level
        top_level_data = dataframes[f'level_{max_level}']
        display_cluster_table(top_level_data, max_level, color_assignments)
    
    else:
        # We've navigated somewhere - figure out what to show
        current_step = st.session_state.navigation_path[-1]
        current_level = current_step['level']
        current_cluster_id = current_step['cluster_id']
        
        if current_level == 0:
            # We're at the bottom level - show individual examples
            display_examples(dataframes['examples'], current_cluster_id)
        else:
            # We're at a middle level - show the child clusters
            current_data = dataframes[f'level_{current_level}']
            current_cluster = current_data[current_data['cluster_id'] == current_cluster_id].iloc[0]
            
            # Find the children of this cluster
            child_cluster_ids = parse_member_clusters(current_cluster['member_clusters'])
            
            if child_cluster_ids:
                # Get the data for the child level
                child_level = current_level - 1
                child_data = dataframes[f'level_{child_level}']
                children = child_data[child_data['cluster_id'].isin(child_cluster_ids)]
                
                display_cluster_table(children, child_level, color_assignments)
            else:
                st.warning("No child clusters found for this selection")

if __name__ == "__main__":
    main()