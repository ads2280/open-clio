import streamlit as st
import pandas as pd
import numpy as np
import ast
import re
from typing import Dict, List, Optional, Tuple

# Page configuration
st.set_page_config(
    page_title="Cluster Explorer",
    page_icon="🔍",
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

def parse_member_clusters(member_str: str) -> List[int]:
    """Parse the member_clusters string to extract cluster IDs"""
    if pd.isna(member_str) or member_str == '':
        return []
    
    try:
        # Handle numpy array string format
        if 'np.int32' in member_str:
            # Extract numbers from numpy array format
            numbers = re.findall(r'np\.int32\((\d+)\)', member_str)
            return [int(num) for num in numbers]
        else:
            # Try to parse as regular list
            return ast.literal_eval(member_str)
    except:
        return []

def get_cluster_colors(num_clusters: int) -> List[str]:
    """Generate distinct colors for clusters"""
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", 
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F8C471", "#82E0AA", "#F1948A", "#85C1E9", "#F4D03F"
    ]
    return colors[:num_clusters] if num_clusters <= len(colors) else colors * (num_clusters // len(colors) + 1)

def display_cluster_table(df: pd.DataFrame, level: int, color_map: Dict[int, str]):
    """Display clusters in a formatted table with click functionality"""
    st.subheader(f"Level {level} Clusters")
    
    # Create clickable table
    for idx, row in df.iterrows():
        cluster_id = row['cluster_id']
        color = color_map.get(cluster_id, "#E8E8E8")
        
        # Create a colored container for each cluster
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 2])
            
            with col1:
                # Color indicator
                st.markdown(f"""
                <div style="width: 30px; height: 30px; background-color: {color}; 
                           border-radius: 50%; margin: 10px;"></div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**{row['name']}**")
                if pd.notna(row['description']):
                    # Clean up description (remove <summary> tags if present)
                    description = str(row['description']).replace('<summary>', '').replace('</summary>', '').strip()
                    st.write(description[:200] + "..." if len(description) > 200 else description)
            
            with col3:
                st.metric("Size", row['size'])
                if 'total_size' in row and pd.notna(row['total_size']) and row['total_size'] != row['size']:
                    st.metric("Total", row['total_size'])
            
            # Make the whole container clickable
            if st.button(f"Explore →", key=f"cluster_{level}_{cluster_id}", use_container_width=True):
                navigate_to_cluster(level, cluster_id, row['name'])
            
            st.divider()

def navigate_to_cluster(level: int, cluster_id: int, cluster_name: str):
    """Navigate to a specific cluster"""
    # Add to navigation path
    st.session_state.navigation_path.append({
        'level': level,
        'cluster_id': cluster_id,
        'name': cluster_name
    })
    st.session_state.selected_cluster = cluster_id
    st.session_state.current_level = level
    st.rerun()

def display_breadcrumb():
    """Display navigation breadcrumb"""
    if not st.session_state.navigation_path:
        return
    
    # Create breadcrumb
    breadcrumb_items = ["🏠 Home"]
    for item in st.session_state.navigation_path:
        breadcrumb_items.append(f"Level {item['level']}: {item['name'][:30]}...")
    
    breadcrumb_text = " → ".join(breadcrumb_items)
    st.markdown(f"**Navigation:** {breadcrumb_text}")
    
    # Back button
    if st.button("← Back"):
        if st.session_state.navigation_path:
            st.session_state.navigation_path.pop()
            if st.session_state.navigation_path:
                last_item = st.session_state.navigation_path[-1]
                st.session_state.current_level = last_item['level']
                st.session_state.selected_cluster = last_item['cluster_id']
            else:
                st.session_state.current_level = None
                st.session_state.selected_cluster = None
        st.rerun()

def display_examples(examples_df: pd.DataFrame, cluster_id: int):
    """Display examples for a base cluster"""
    cluster_examples = examples_df[examples_df['base_cluster_id'] == cluster_id]
    
    st.subheader(f"Examples in Cluster (Total: {len(cluster_examples)})")
    
    # Pagination
    items_per_page = 10
    total_pages = (len(cluster_examples) - 1) // items_per_page + 1
    
    if total_pages > 1:
        page = st.selectbox("Page", range(1, total_pages + 1)) - 1
    else:
        page = 0
    
    start_idx = page * items_per_page
    end_idx = min(start_idx + items_per_page, len(cluster_examples))
    
    # Display examples
    for idx, row in cluster_examples.iloc[start_idx:end_idx].iterrows():
        with st.expander(f"Example {idx + 1}: {row['example_id'][:8]}..."):
            st.write("**Summary:**")
            st.write(row['summary'])
            st.write(f"**Example ID:** `{row['example_id']}`")

def main():
    st.title("🔍 Hierarchical Cluster Explorer")
    st.markdown("Navigate through your cluster hierarchy and explore conversation examples")
    
    init_session_state()
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File uploaders
        examples_file = st.file_uploader("Examples CSV", type=['csv'], key="examples")
        level_0_file = st.file_uploader("Level 0 Clusters CSV", type=['csv'], key="level0")
        level_1_file = st.file_uploader("Level 1 Clusters CSV", type=['csv'], key="level1")
        
        # Optional higher levels
        st.markdown("**Optional Higher Levels:**")
        level_2_file = st.file_uploader("Level 2 Clusters CSV", type=['csv'], key="level2")
        level_3_file = st.file_uploader("Level 3 Clusters CSV", type=['csv'], key="level3")
        
        if st.button("Load Data"):
            files_dict = {
                'examples': examples_file,
                'level_0': level_0_file,
                'level_1': level_1_file,
                'level_2': level_2_file,
                'level_3': level_3_file
            }
            
            # Filter out None files
            files_dict = {k: v for k, v in files_dict.items() if v is not None}
            
            if 'examples' in files_dict and 'level_0' in files_dict:
                try:
                    dataframes = load_csv_files(files_dict)
                    st.session_state.dataframes = dataframes
                    st.session_state.data_loaded = True
                    st.session_state.max_level = max([int(k.split('_')[1]) for k in dataframes.keys() if 'level_' in k])
                    st.success("Data loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            else:
                st.warning("Please upload at least the Examples and Level 0 files")
    
    # Main content area
    if not st.session_state.data_loaded:
        st.info("👆 Please upload your CSV files in the sidebar to get started")
        return
    
    # Display breadcrumb navigation
    display_breadcrumb()
    
    dataframes = st.session_state.dataframes
    
    # Determine what to show based on navigation state
    if not st.session_state.navigation_path:
        # Show highest level clusters
        max_level = st.session_state.max_level
        df = dataframes[f'level_{max_level}']
        colors = get_cluster_colors(len(df))
        color_map = {}
        for idx, (_, row) in enumerate(df.iterrows()):
            color_map[row['cluster_id']] = colors[idx % len(colors)]
        display_cluster_table(df, max_level, color_map)
    
    else:
        # We're navigating down the hierarchy
        current_item = st.session_state.navigation_path[-1]
        current_level = current_item['level']
        current_cluster_id = current_item['cluster_id']
        
        if current_level == 0:
            # We're at base level - show examples
            display_examples(dataframes['examples'], current_cluster_id)
        else:
            # Show child clusters
            current_df = dataframes[f'level_{current_level}']
            current_row = current_df[current_df['cluster_id'] == current_cluster_id].iloc[0]
            
            # Parse member clusters
            member_cluster_ids = parse_member_clusters(current_row['member_clusters'])
            
            if member_cluster_ids:
                # Get child level data
                child_level = current_level - 1
                child_df = dataframes[f'level_{child_level}']
                child_clusters = child_df[child_df['cluster_id'].isin(member_cluster_ids)]
                
                colors = get_cluster_colors(len(child_clusters))
                color_map = {}
                for idx, (_, row) in enumerate(child_clusters.iterrows()):
                    color_map[row['cluster_id']] = colors[idx % len(colors)]
                
                display_cluster_table(child_clusters, child_level, color_map)
            else:
                st.warning("No child clusters found for this selection")

if __name__ == "__main__":
    main()