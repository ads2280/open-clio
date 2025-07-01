import streamlit as st
import pandas as pd
import ast
import re
from typing import Dict, List

# Page config
st.set_page_config(
    page_title="Cluster Explorer", layout="wide", initial_sidebar_state="expanded"
)


# Initialise state
def init_session_state():
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "navigation_path" not in st.session_state:
        st.session_state.navigation_path = []
    if "current_level" not in st.session_state:
        st.session_state.current_level = None
    if "selected_cluster" not in st.session_state:
        st.session_state.selected_cluster = None
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "clusters"
    if "selected_example_id" not in st.session_state:
        st.session_state.selected_example_id = None


# Load and cache CSVs
@st.cache_data
def load_csv_files(files_dict: Dict[str, any]) -> Dict[str, pd.DataFrame]:
    dataframes = {}

    for file_type, uploaded_file in files_dict.items():
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            dataframes[file_type] = df

    return dataframes


# Parse member_clusters from list str to int
def parse_member_clusters(member_str: str) -> List[int]:
    if pd.isna(member_str) or member_str == "":
        return []
    try:
        if "np.int32" in member_str:
            numbers = re.findall(r"np\.int32\((\d+)\)", member_str)
            return [int(num) for num in numbers]
        else:
            return ast.literal_eval(member_str)
    except:
        return []


# Colours for top level clusters
def get_distinct_colours():
    return [
        "#DC2626",
        "#059669",
        "#2563EB",
        "#7C3AED",
        "#EA580C",
        "#0891B2",
        "#BE185D",
        "#65A30D",
    ]


# Take one colour and create multiple lighter versions of it
def create_colour_variations(base_colour, num_variations):
    variations = []

    # Remove the # symbol and convert hex to RGB numbers
    hex_colour = base_colour.replace("#", "")
    red = int(hex_colour[0:2], 16)
    green = int(hex_colour[2:4], 16)
    blue = int(hex_colour[4:6], 16)

    for i in range(num_variations):
        if i == 0:
            # First one is the original colour
            variations.append(base_colour)
        else:
            # Make each subsequent colour lighter by mixing with white (255)
            lightness = 0.3 + (i * 0.4 / (num_variations - 1))

            new_red = int(red + (255 - red) * lightness)
            new_green = int(green + (255 - green) * lightness)
            new_blue = int(blue + (255 - blue) * lightness)

            # Convert back to hex
            lighter_colour = f"#{new_red:02x}{new_green:02x}{new_blue:02x}"
            variations.append(lighter_colour)

    return variations


# assign colours so each top level cluster gets a distinct one, and each child gets a related shade
def assign_colours_to_clusters(dataframes, max_level):
    colour_assignments = {}
    distinct_colours = get_distinct_colours()

    # Start with top lvl clusters, go through each
    top_level_clusters = dataframes[f"level_{max_level}"]

    for i, (_, cluster_row) in enumerate(top_level_clusters.iterrows()):
        cluster_id = cluster_row["cluster_id"]

        # Pick a distinct colour
        base_colour = distinct_colours[i % len(distinct_colours)]
        colour_assignments[(max_level, cluster_id)] = base_colour

        # Set colours for all children of thiscluster
        if max_level > 0:  # Only if there are child levels
            child_ids = parse_member_clusters(cluster_row["member_clusters"])

            if child_ids:
                # Variation on parent colour
                child_colours = create_colour_variations(base_colour, len(child_ids))
                for child_index, child_id in enumerate(child_ids):
                    colour_assignments[(max_level - 1, child_id)] = child_colours[
                        child_index
                    ]

    return colour_assignments


# Display clusters
def display_cluster_table(df: pd.DataFrame, level: int, colour_assignments: Dict):
    st.subheader(f"Level {level} Clusters")

    # Show each cluster as a row
    for _, cluster_row in df.iterrows():
        cluster_id = cluster_row["cluster_id"]

        cluster_colour = colour_assignments.get((level, cluster_id), "#9CA3AF")
        with st.container():
            col1, col2, col3 = st.columns([1, 6, 2])

            # Coloured circle
            with col1:
                st.markdown(
                    f"""
                <div style="width: 30px; height: 30px; background-color: {cluster_colour}; 
                           border-radius: 50%; margin: 10px;"></div>
                """,
                    unsafe_allow_html=True,
                )

            # Cluster name and description in middle column
            with col2:
                st.markdown(f"**{cluster_row['name']}**")
                if pd.notna(cluster_row["description"]):
                    # Clean up the description text
                    description = (
                        str(cluster_row["description"])
                        .replace("<summary>", "")
                        .replace("</summary>", "")
                        .strip()
                    )
                    shortened = (
                        description[:200] + "..."
                        if len(description) > 200
                        else description
                    )
                    st.write(shortened)

            # Metrics in last column
            with col3:
                st.metric("Size", cluster_row["size"])
                if (
                    "total_size" in cluster_row
                    and pd.notna(cluster_row["total_size"])
                    and cluster_row["total_size"] != cluster_row["size"]
                ):
                    st.metric("Total", cluster_row["total_size"])

            # Explore button
            if st.button(
                "Explore →",
                key=f"cluster_{level}_{cluster_id}",
                use_container_width=True,
            ):
                navigate_to_cluster(level, cluster_id, cluster_row["name"])

            st.divider()


def navigate_to_cluster(level: int, cluster_id: int, cluster_name: str):
    """Add this cluster to our navigation path and refresh the page"""
    st.session_state.navigation_path.append(
        {"level": level, "cluster_id": cluster_id, "name": cluster_name}
    )
    st.session_state.selected_cluster = cluster_id
    st.session_state.current_level = level
    st.rerun()


def display_breadcrumb():
    """Show where we are in the navigation"""
    if not st.session_state.navigation_path:
        return

    # Build the breadcrumb trail
    trail = ["Home"]
    for step in st.session_state.navigation_path:
        short_name = (
            step["name"][:50] + "..." if len(step["name"]) > 50 else step["name"]
        )
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
                st.session_state.current_level = last_step["level"]
                st.session_state.selected_cluster = last_step["cluster_id"]
            else:
                # Back to the beginning
                st.session_state.current_level = None
                st.session_state.selected_cluster = None
        st.rerun()


def display_examples(examples_df: pd.DataFrame, cluster_id: int):
    """Show the actual conversation examples for a base cluster"""
    # Filter to just the examples in this cluster
    cluster_examples = examples_df[examples_df["base_cluster_id"] == cluster_id]

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
    for i, (original_index, example_row) in enumerate(
        cluster_examples.iloc[start:end].iterrows()
    ):
        example_id = example_row["example_id"]

        with st.container():
            col1, col2 = st.columns([8, 2])

            with col1:
                st.markdown(f"**Example {start + i + 1}:** `{example_id}`")
                st.write(f"**Summary:** {example_row['summary']}")

            with col2:
                if st.button(
                    "View Full", key=f"view_full_{example_id}", use_container_width=True
                ):
                    # Store both the example_id and the original row index for matching
                    st.session_state.selected_example_id = example_id
                    st.session_state.selected_example_index = original_index
                    st.session_state.view_mode = "full_example"
                    st.rerun()

            st.divider()


def display_full_example(full_examples_df: pd.DataFrame, example_id: str):
    """Show the complete conversation data for a specific example"""

    # Get the row index we stored when the user clicked "View Full"
    if "selected_example_index" not in st.session_state:
        st.error("No example index found. Please go back and select an example again.")
        return

    example_index = st.session_state.selected_example_index

    # Check if the index is valid
    if example_index >= len(full_examples_df):
        st.error(
            f"Example index {example_index} is out of range. Full examples CSV has {len(full_examples_df)} rows."
        )
        return

    # Get the row by index
    example_row = full_examples_df.iloc[example_index]

    # Header with back button
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("← Back to Examples", type="secondary"):
            st.session_state.view_mode = "clusters"
            st.session_state.selected_example_id = None
            st.session_state.selected_example_index = None
            st.rerun()

    with col2:
        st.subheader(f"Full Example: {example_id}")

    # Display the conversation
    st.markdown(
        f"**Example ID:** `{example_id}` (Row {example_index} in full examples CSV)"
    )

    # Show Input Query first
    if "input_query" in example_row.index and pd.notna(example_row["input_query"]):
        st.markdown("**Input Query:**")
        st.write(example_row["input_query"])
        st.divider()

    # Show Input Answer second
    if "input_answer" in example_row.index and pd.notna(example_row["input_answer"]):
        st.markdown("**Input Answer:**")
        st.write(example_row["input_answer"])


def main():
    st.title("Hierarchical Cluster Explorer")
    st.markdown(
        "Navigate through your cluster hierarchy and explore conversation examples"
    )

    init_session_state()

    # Sidebar for uploading files
    with st.sidebar:
        st.header("Data Upload")

        # Required files
        full_examples_file = st.file_uploader(
            "Full Examples CSV (complete data)", type=["csv"], key="full_examples"
        )
        examples_file = st.file_uploader(
            "Examples CSV (summaries)", type=["csv"], key="examples"
        )
        level_0_file = st.file_uploader(
            "Level 0 Clusters CSV", type=["csv"], key="level0"
        )
        level_1_file = st.file_uploader(
            "Level 1 Clusters CSV", type=["csv"], key="level1"
        )

        # Optional additional levels
        st.markdown("**Optional Higher Levels:**")
        level_2_file = st.file_uploader(
            "Level 2 Clusters CSV", type=["csv"], key="level2"
        )
        level_3_file = st.file_uploader(
            "Level 3 Clusters CSV", type=["csv"], key="level3"
        )

        # Load button
        if st.button("Load Data"):
            files = {
                "examples": examples_file,
                "full_examples": full_examples_file,
                "level_0": level_0_file,
                "level_1": level_1_file,
                "level_2": level_2_file,
                "level_3": level_3_file,
            }

            # Remove empty file slots
            files = {name: file for name, file in files.items() if file is not None}

            # Check we have the minimum required files
            if "examples" in files and "full_examples" in files and "level_0" in files:
                try:
                    # Load all the CSV files
                    dataframes = load_csv_files(files)
                    st.session_state.dataframes = dataframes
                    st.session_state.data_loaded = True

                    # Figure out what the highest level is
                    level_numbers = [
                        int(name.split("_")[1])
                        for name in dataframes.keys()
                        if "level_" in name
                    ]
                    st.session_state.max_level = max(level_numbers)

                    st.success("Data loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            else:
                st.warning(
                    "Please upload at least the Examples, Full Examples, and Level 0 files"
                )

    # Main area
    if not st.session_state.data_loaded:
        st.info("Uppload your CSV files in the sidebar to get started")

        # Show example of expected data format
        with st.expander("Expected Data Format"):
            st.markdown("""
            **Examples CSV (summaries) should have columns:**
            - `example_id`: Unique identifier for each conversation
            - `summary`: Text summary of the conversation
            - `base_cluster_id`: ID of the base cluster this example belongs to
            - `base_cluster_name`: Name of the base cluster
            - `base_cluster_description`: Description of the base cluster
            
            **Full Examples CSV (complete data) should have columns:**
            - `example_id`: Same IDs as in the Examples CSV
            - Plus any additional columns with the full conversation data (e.g., `input_messages`, `input_query`, `output_request`, etc.)
            
            **Cluster CSVs should have columns:**
            - `cluster_id`: Unique identifier for the cluster
            - `name`: Cluster name
            - `description`: Cluster description
            - `size`: Number of direct members
            - `total_size`: Total number including sub-clusters
            - `member_clusters`: List of child cluster IDs (for hierarchical levels)
            """)
        return

    # Show the navigation breadcrumb
    display_breadcrumb()

    dataframes = st.session_state.dataframes

    # Create the colour scheme once and store it
    if "colour_assignments" not in st.session_state:
        st.session_state.colour_assignments = assign_colours_to_clusters(
            dataframes, st.session_state.max_level
        )

    colour_assignments = st.session_state.colour_assignments

    # Decide what to show based on where we are in the navigation
    if st.session_state.view_mode == "full_example":
        # Show full example details
        display_full_example(
            dataframes["full_examples"], st.session_state.selected_example_id
        )

    elif not st.session_state.navigation_path:
        # We're at top, show the highest level clusters
        max_level = st.session_state.max_level
        top_level_data = dataframes[f"level_{max_level}"]
        display_cluster_table(top_level_data, max_level, colour_assignments)

    else:
        # We've navigated, now figure out what to show
        current_step = st.session_state.navigation_path[-1]
        current_level = current_step["level"]
        current_cluster_id = current_step["cluster_id"]

        if current_level == 0:
            # We're at the bottom level - show individual examples
            display_examples(dataframes["examples"], current_cluster_id)
        else:
            # We're at a middle level, show its child clusters
            current_data = dataframes[f"level_{current_level}"]
            current_cluster = current_data[
                current_data["cluster_id"] == current_cluster_id
            ].iloc[0]

            # Find the children of this cluster
            child_cluster_ids = parse_member_clusters(
                current_cluster["member_clusters"]
            )

            if child_cluster_ids:
                # Get the data for the child level
                child_level = current_level - 1
                child_data = dataframes[f"level_{child_level}"]
                children = child_data[child_data["cluster_id"].isin(child_cluster_ids)]

                display_cluster_table(children, child_level, colour_assignments)
            else:
                st.warning("No child clusters found for this selection")


if __name__ == "__main__":
    main()
