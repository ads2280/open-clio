import ast
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Clustering Explorer", layout="wide", initial_sidebar_state="expanded"
)


class ClusteringExplorer:
    def __init__(self, config_path: str = ".data/config.json"):
        self.config = self.load_config(config_path)
        self.data = {}
        self.max_level = len(self.config["hierarchy"]) - 1
        self.level_names = [f"level_{i}" for i in range(len(self.config["hierarchy"]))]
        # Use the same save path as generate.py
        self.save_path = ".data/clustering_results"

    def load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            st.error(f"Config file {config_path} not found!")
            st.stop()
        except json.JSONDecodeError:
            st.error(f"Invalid JSON in {config_path}")
            st.stop()

    @st.cache_data
    def load_data(_self) -> Dict[str, pd.DataFrame]:
        """Load combined.csv and level data files"""
        data = {}
        save_path = _self.save_path

        # Load combined.csv - required
        combined_path = os.path.join(save_path, "combined.csv")
        if os.path.exists(combined_path):
            try:
                data["combined"] = pd.read_csv(combined_path)
                st.success(
                    f"Loaded combined dataset with {len(data['combined'])} examples"
                )
            except Exception as e:
                st.error(f"Error loading combined.csv: {e}")
                st.stop()
        else:
            st.error(f"combined.csv not found in {save_path}")
            st.error("Please run generate_clusters.py to generate cluster info")
            st.stop()

        # Load level data files
        for i in range(_self.max_level + 1):
            level_name = f"level_{i}"
            level_path = os.path.join(save_path, f"{level_name}_clusters.csv")
            if os.path.exists(level_path):
                try:
                    data[level_name] = pd.read_csv(level_path)
                    st.success(
                        f"Loaded {level_name} with {len(data[level_name])} clusters"
                    )
                except Exception as e:
                    st.error(f"Error loading {level_name}_clusters.csv: {e}")
                    st.stop()
            else:
                st.warning(f"{level_name}_clusters.csv not found in {save_path}")

        return data

    def parse_member_clusters(self, member_str: str) -> List[int]:
        """Parse member_clusters from string to list of ints"""
        if pd.isna(member_str) or member_str == "":
            return []
        elif "np.int32" in str(member_str):
            numbers = re.findall(r"np\.int32\((\d+)\)", str(member_str))
            return [int(num) for num in numbers]
        else:
            return ast.literal_eval(str(member_str))

    def get_distinct_colors(self, n_colors: int) -> List[str]:
        """Generate distinct colors for clusters"""
        base_colors = [
            "#DC2626",
            "#059669",
            "#2563EB",
            "#7C3AED",
            "#EA580C",
            "#0891B2",
            "#BE185D",
            "#65A30D",
            "#F59E0B",
            "#EF4444",
            "#10B981",
            "#3B82F6",
            "#8B5CF6",
            "#F97316",
            "#06B6D4",
            "#EC4899",
            "#84CC16",
            "#FCD34D",
            "#F87171",
            "#34D399",
        ]

        colors = []
        for i in range(n_colors):
            colors.append(base_colors[i % len(base_colors)])
        return colors

    def create_color_variations(
        self, base_color: str, num_variations: int
    ) -> List[str]:
        """Create lighter variations of a base color"""
        variations = [base_color]

        if num_variations <= 1:
            return variations

        hex_color = base_color.replace("#", "")
        r, g, b = (
            int(hex_color[0:2], 16),
            int(hex_color[2:4], 16),
            int(hex_color[4:6], 16),
        )

        for i in range(1, num_variations):
            lightness = 0.3 + (i * 0.4 / (num_variations - 1))
            new_r = int(r + (255 - r) * lightness)
            new_g = int(g + (255 - g) * lightness)
            new_b = int(b + (255 - b) * lightness)
            variations.append(f"#{new_r:02x}{new_g:02x}{new_b:02x}")

        return variations

    def assign_colors_to_clusters(self) -> Dict[Tuple[int, int], str]:
        """Assign colors to all clusters across all levels"""
        if f"level_{self.max_level}" not in self.data:
            return {}

        color_assignments = {}
        distinct_colors = self.get_distinct_colors(20)

        top_level_data = self.data[
            f"level_{self.max_level}"
        ]  # start with top level clusters

        for i, (_, cluster_row) in enumerate(top_level_data.iterrows()):
            cluster_id = str(cluster_row["cluster_id"])
            base_color = distinct_colors[i % len(distinct_colors)]
            color_assignments[(self.max_level, cluster_id)] = base_color

            self._assign_child_colors(
                self.max_level, cluster_id, base_color, color_assignments
            )  # Recursively assign colors to child clusters

        return color_assignments

    def _assign_child_colors(
        self, level: int, cluster_id: int, base_color: str, color_assignments: Dict
    ):
        """Recursively assign colors to child clusters"""
        if level == 0:
            return

        current_data = self.data.get(f"level_{level}")  # get current cluster data
        if current_data is None:
            return

        current_cluster = current_data[current_data["cluster_id"] == cluster_id]
        if len(current_cluster) == 0:
            return

        current_cluster = current_cluster.iloc[0]
        child_cluster_ids = self.parse_member_clusters(
            current_cluster.get("member_clusters", "")
        )

        if child_cluster_ids:
            child_colors = self.create_color_variations(
                base_color, len(child_cluster_ids)
            )

            for child_index, child_id in enumerate(child_cluster_ids):
                child_color = child_colors[child_index % len(child_colors)]
                color_assignments[(level - 1, child_id)] = child_color

                self._assign_child_colors(
                    level - 1, child_id, child_color, color_assignments
                )

    def get_child_clusters(self, level: int, cluster_id: int) -> pd.DataFrame:
        """Get a cluster's child clusters"""
        if level == 0:
            return pd.DataFrame()

        current_data = self.data.get(f"level_{level}")
        if current_data is None:
            return pd.DataFrame()

        current_cluster = current_data[current_data["cluster_id"] == cluster_id]
        if len(current_cluster) == 0:
            return pd.DataFrame()

        current_cluster = current_cluster.iloc[0]
        child_cluster_ids = self.parse_member_clusters(
            current_cluster.get("member_clusters", "")
        )

        if not child_cluster_ids:
            return pd.DataFrame()

        child_level = level - 1
        child_data = self.data.get(f"level_{child_level}")
        if child_data is None:
            return pd.DataFrame()

        return child_data[child_data["cluster_id"].isin(child_cluster_ids)]


def init_session_state(explorer: ClusteringExplorer):
    """Set state variables"""
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
        try:
            explorer.data = explorer.load_data()
            st.session_state.data_loaded = True
            st.session_state.max_level = explorer.max_level
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

    if "navigation_path" not in st.session_state:
        st.session_state.navigation_path = []
    if "current_level" not in st.session_state:
        st.session_state.current_level = None
    if "selected_cluster" not in st.session_state:
        st.session_state.selected_cluster = None
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = (
            "partitions" if explorer.config.get("partitions") else "clusters"
        )
    if "selected_partition" not in st.session_state:
        st.session_state.selected_partition = None
    if "selected_example_id" not in st.session_state:
        st.session_state.selected_example_id = None
    if "color_assignments" not in st.session_state:
        st.session_state.color_assignments = explorer.assign_colors_to_clusters()


def display_cluster_table(
    df: pd.DataFrame, level: int, color_assignments: Dict, explorer: ClusteringExplorer
):
    """Display clusters in UI"""
    st.subheader(f"Level {level} Clusters")

    for _, cluster_row in df.iterrows():
        cluster_id = cluster_row["cluster_id"]
        cluster_color = color_assignments.get((level, cluster_id), "#9CA3AF")

        with st.container():
            col1, col2, col3 = st.columns([1, 6, 2])

            with col1:
                st.markdown(
                    f"""
                <div style="width: 30px; height: 30px; background-color: {cluster_color}; 
                           border-radius: 50%; margin: 10px;"></div>
                """,
                    unsafe_allow_html=True,
                )

            # Cluster name and description
            with col2:
                st.markdown(f"**{cluster_row['name']}**")

                partition_col = None
                for col in ["category", "partition"]:  # shows partition if it exists
                    if col in cluster_row.index and pd.notna(cluster_row[col]):
                        partition_col = col
                        break

                if partition_col:
                    st.markdown(
                        f"*{partition_col.title()}: {cluster_row[partition_col]}*"
                    )

                if pd.notna(cluster_row.get("description")):
                    description = (
                        str(cluster_row["description"])
                        .replace("<summary>", "")
                        .replace("</summary>", "")
                        .strip()
                    )
                    st.write(description)

            # Metrics
            with col3:
                if level > 0:
                    st.metric("Sub-clusters", cluster_row.get("size", 0))
                    if "total_size" in cluster_row and pd.notna(
                        cluster_row["total_size"]
                    ):
                        st.metric("Total items", cluster_row["total_size"])
                else:
                    st.metric("Items", cluster_row.get("size", 0))

            # Navigation button
            if st.button(
                "Explore →",
                key=f"cluster_{level}_{cluster_id}",
                use_container_width=True,
            ):
                navigate_to_cluster(level, cluster_id, cluster_row["name"])

            st.divider()


def navigate_to_cluster(level: int, cluster_id: int, cluster_name: str):
    """Navigate to a cluster"""
    st.session_state.navigation_path.append(
        {"level": level, "cluster_id": cluster_id, "name": cluster_name}
    )
    st.session_state.selected_cluster = cluster_id
    st.session_state.current_level = level

    # Determine what to show next
    if level == 0:
        st.session_state.view_mode = "examples"
    else:
        st.session_state.view_mode = "clusters"

    st.rerun()


def display_parent_stack():
    """Show navigation breadcrumb"""
    if not st.session_state.navigation_path:
        return

    # Back button
    if st.button("← Back"):
        if st.session_state.navigation_path:
            st.session_state.navigation_path.pop()

            if st.session_state.navigation_path:
                last_step = st.session_state.navigation_path[-1]
                st.session_state.current_level = last_step["level"]
                st.session_state.selected_cluster = last_step["cluster_id"]
                st.session_state.view_mode = (
                    "examples" if last_step["level"] == 0 else "clusters"
                )
            else:
                # Back to top level
                st.session_state.current_level = None
                st.session_state.selected_cluster = None
                st.session_state.view_mode = (
                    "partitions"
                    if st.session_state.get("has_partitions")
                    else "clusters"
                )
        st.rerun()

    # Show breadcrumb
    for step in st.session_state.navigation_path:
        st.markdown(f"**L{step['level']}: {step['name']}**")

    # Show current cluster description
    if st.session_state.navigation_path:
        current_step = st.session_state.navigation_path[-1]
        current_level = current_step["level"]
        current_cluster_id = current_step["cluster_id"]

        # Get cluster data
        explorer = st.session_state.explorer
        current_data = explorer.data.get(f"level_{current_level}")
        if current_data is not None:
            current_cluster = current_data[
                current_data["cluster_id"] == current_cluster_id
            ]
            if len(current_cluster) > 0:
                current_cluster = current_cluster.iloc[0]
                if pd.notna(current_cluster.get("description")):
                    description = (
                        str(current_cluster["description"])
                        .replace("<summary>", "")
                        .replace("</summary>", "")
                        .strip()
                    )
                    st.markdown(f"{description}")

        st.markdown("---")


def display_examples(explorer: ClusteringExplorer, cluster_id: int):
    """Display examples for a base-level cluster"""
    combined_data = explorer.data.get("combined")
    if combined_data is None:
        st.error("No combined data available to show examples")
        return

    # Filter examples for this cluster
    cluster_examples = combined_data[combined_data["base_cluster_id"] == cluster_id]

    # Show cluster info
    level_0_data = explorer.data.get("level_0")
    if level_0_data is not None:
        current_cluster = level_0_data[level_0_data["cluster_id"] == cluster_id]
        if len(current_cluster) > 0:
            partition_col = None
            for col in ["category", "partition"]:
                if col in current_cluster.columns and pd.notna(
                    current_cluster.iloc[0].get(col)
                ):
                    partition_col = col
                    break

            if partition_col:
                st.markdown(
                    f"**{partition_col.title()}:** {current_cluster.iloc[0][partition_col]}"
                )

    st.subheader(f"Examples in Cluster (Total: {len(cluster_examples)})")

    examples_per_page = 10
    total_pages = (len(cluster_examples) - 1) // examples_per_page + 1

    if total_pages > 1:
        page = st.selectbox("Page", range(1, total_pages + 1)) - 1
    else:
        page = 0

    start = page * examples_per_page
    end = min(start + examples_per_page, len(cluster_examples))

    # Display examples
    for i, (_, example_row) in enumerate(cluster_examples.iloc[start:end].iterrows()):
        example_id = example_row["example_id"]

        with st.container():
            col1, col2 = st.columns([8, 2])

            with col1:
                st.markdown(f"**Example {start + i + 1}:** `{example_id}`")
                st.write(f"**Summary:** {example_row['summary']}")

                # Show any additional columns
                for col in example_row.index:
                    if col not in [
                        "example_id",
                        "summary",
                        "base_cluster_id",
                        "base_cluster_name",
                    ] and pd.notna(example_row[col]):
                        st.write(f"**{col.title()}:** {example_row[col]}")

            with col2:
                if "full_example" in example_row.index and pd.notna(
                    example_row["full_example"]
                ):
                    if st.button(
                        "View Full",
                        key=f"view_full_{example_id}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_example_id = example_id
                        st.session_state.view_mode = "full_example"
                        st.rerun()

            st.divider()


def display_full_example(combined_data: pd.DataFrame, example_id: str):
    """Display full example details"""
    matching_examples = combined_data[combined_data["example_id"] == example_id]

    if len(matching_examples) == 0:
        st.error(f"Example {example_id} not found.")
        return

    example_row = matching_examples.iloc[0]

    # Header with back button
    col1, col2 = st.columns([1, 8])
    with col1:
        if st.button("← Back to Examples", type="secondary"):
            st.session_state.view_mode = "examples"
            st.session_state.selected_example_id = None
            st.rerun()

    with col2:
        st.subheader(f"Example {example_id}")

    # Display all available information
    for col in example_row.index:
        if pd.notna(example_row[col]) and col != "example_id":
            st.markdown(f"**{col.replace('_', ' ').title()}:** {example_row[col]}")


def build_hierarchy_display(explorer: ClusteringExplorer) -> str:
    """Build a nice hierarchy display string"""
    config_hierarchy = explorer.config["hierarchy"]

    # check what levels actually exist
    actual_levels = []
    for i, expected_count in enumerate(config_hierarchy):
        level_data = explorer.data.get(f"level_{i}")
        if level_data is not None:
            actual_levels.append(len(level_data))
        else:
            break

    if len(actual_levels) == 1:
        return f"{actual_levels[0]} clusters (single level)"
    else:
        return " → ".join(map(str, actual_levels)) + " clusters"


def show_cluster_previews_in_sidebar(explorer: ClusteringExplorer):
    """Show cluster name previews in sidebar"""
    available_levels = []
    for i, level_name in enumerate(explorer.level_names):
        level_data = explorer.data.get(level_name)
        if level_data is not None:
            available_levels.append((i, level_data))

    if not available_levels:
        return

    max_level = max([level for level, _ in available_levels])

    for level, level_data in available_levels:
        if level == 0:
            label = "(Base)"
        elif level == max_level:
            label = "(Top)"
        else:
            label = ""

        st.markdown(f"### Level {level} Clusters {label}")

        # Show top 3-4 cluster names
        preview_count = min(4, len(level_data))
        for i, (_, cluster_row) in enumerate(level_data.head(preview_count).iterrows()):
            # Truncate long names
            name = cluster_row["name"]
            if len(name) > 50:
                name = name[:47] + "..."
            st.markdown(f"- {name}")

        if len(level_data) > preview_count:
            if st.button(
                f"Show all {len(level_data)} clusters ↓", key=f"show_all_{level}"
            ):
                # Navigate to this level
                if level == max_level:
                    st.session_state.view_mode = "clusters"
                    st.session_state.navigation_path = []
                else:
                    # This is trickier - would need to implement level-specific views
                    st.info(
                        f"Navigate through the hierarchy to see Level {level} clusters"
                    )
                st.rerun()

        st.markdown("")


def display_partition_overview(explorer: ClusteringExplorer):
    """Show partition/category overview as landing page"""
    partitions = explorer.config.get("partitions", {})
    if not partitions:
        # No partitions defined, go straight to clusters
        st.session_state.view_mode = "clusters"
        st.rerun()
        return

    st.header("Dataset Partitions")

    # Calculate stats by partition
    level_0_data = explorer.data.get("level_0")
    if level_0_data is None:
        st.error("No level 0 data available")
        return

    # Find partition column
    partition_col = None
    for col in ["category", "partition"]:
        if col in level_0_data.columns:
            partition_col = col
            break

    if partition_col is None:
        st.warning("No partition column found in data")
        st.session_state.view_mode = "clusters"
        st.rerun()
        return

    partition_stats = (
        level_0_data.groupby(partition_col).agg({"size": ["count", "sum"]}).round()
    )
    partition_stats.columns = ["clusters", "total_items"]
    partition_stats = partition_stats.sort_values("total_items", ascending=False)

    # Display as cards
    cols = st.columns(2)
    for i, (partition_name, stats) in enumerate(partition_stats.iterrows()):
        with cols[i % 2]:
            with st.container():
                st.markdown(f"### {partition_name}")

                # Show description if available
                if partition_name in partitions:
                    st.markdown(f"{partitions[partition_name]}")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Clusters", int(stats["clusters"]))
                with col2:
                    st.metric("Items", int(stats["total_items"]))

                if st.button(
                    f"Explore {partition_name} →",
                    key=f"partition_overview_{i}",
                    use_container_width=True,
                ):
                    st.session_state.selected_partition = partition_name
                    st.session_state.view_mode = "partition_clusters"
                    st.rerun()

                st.markdown("---")
    """Build a nice hierarchy display string"""
    config_hierarchy = explorer.config["hierarchy"]

    # Check what levels actually exist in the data
    actual_levels = []
    for i, expected_count in enumerate(config_hierarchy):
        level_data = explorer.data.get(f"level_{i}")
        if level_data is not None:
            actual_levels.append(len(level_data))
        else:
            break

    if len(actual_levels) == 1:
        return f"{actual_levels[0]} clusters (single level)"
    else:
        return " → ".join(map(str, actual_levels)) + " clusters"


def display_partition_clusters(explorer: ClusteringExplorer, partition_name: str):
    """Show clusters for a specific partition"""
    # Back button
    if st.button("← Back to Partitions"):
        st.session_state.selected_partition = None
        st.session_state.view_mode = "partitions"
        st.rerun()

    st.subheader(f"{partition_name} - Clusters")

    # Filter clusters to partition
    level_0_data = explorer.data.get("level_0")
    if level_0_data is None:
        st.error("No level 0 data available")
        return

    partition_col = None
    for col in ["category", "partition"]:
        if col in level_0_data.columns:
            partition_col = col
            break

    if partition_col is None:
        st.error("No partition column found")
        return

    partition_data = level_0_data[level_0_data[partition_col] == partition_name]
    partition_data = partition_data.sort_values("size", ascending=False)

    display_cluster_table(
        partition_data, 0, st.session_state.color_assignments, explorer
    )


def main():
    st.title("OpenCLIO Clustering Explorer")

    # Initialize explorer
    if "explorer" not in st.session_state:
        st.session_state.explorer = ClusteringExplorer()

    explorer = st.session_state.explorer
    init_session_state(explorer)

    if not st.session_state.data_loaded:
        st.error("Failed to load data. Please check your configuration and data files.")
        return

    # Show dataset info with better hierarchy display
    hierarchy_display = build_hierarchy_display(explorer)
    st.markdown(f"**Dataset:** {explorer.config['dataset_name']}")
    st.markdown(f"**Hierarchy:** {hierarchy_display}")

    # Sidebar with enhanced dataset summary
    with st.sidebar:
        st.header("Dataset Overview")

        # Calculate total items
        level_0_data = explorer.data.get("level_0")
        if level_0_data is not None:
            total_items = level_0_data["size"].sum()
            st.metric("Total Items", f"{total_items:,}")

            # Show partitions if available
            partitions = explorer.config.get("partitions")
            if partitions and level_0_data is not None:
                st.divider()
                st.markdown("### Partitions")
                st.markdown(
                    "*Partitions group the data by main topic areas or categories*"
                )

                partition_col = None
                for col in ["category", "partition"]:
                    if col in level_0_data.columns:
                        partition_col = col
                        break

                if partition_col:
                    partition_counts = (
                        level_0_data.groupby(partition_col)["size"]
                        .sum()
                        .sort_values(ascending=False)
                    )
                    for partition, count in partition_counts.items():
                        percentage = (count / total_items) * 100
                        st.markdown(
                            f"**{partition}**: {count:,} items ({percentage:.1f}%)"
                        )
            st.divider()
            # Show hierarchy breakdown with labels
            st.markdown("### Hierarchy Levels")
            available_levels = []
            for i, level_name in enumerate(explorer.level_names):
                level_data = explorer.data.get(level_name)
                if level_data is not None:
                    available_levels.append((i, len(level_data)))

            if available_levels:
                max_level = max([level for level, _ in available_levels])
                for level, count in available_levels:
                    if level == 0:
                        label = "(Base)"
                    elif level == max_level:
                        label = "(Top)"
                    else:
                        label = ""
                    st.markdown(f"**Level {level} {label}:** {count} clusters")

        # Show cluster previews for each level
        show_cluster_previews_in_sidebar(explorer)

    # Show navigation breadcrumb
    display_parent_stack()

    # Decide what content to show based on what mode you're viewing in
    if st.session_state.view_mode == "full_example":
        combined_data = explorer.data.get("combined")
        if combined_data is not None:
            display_full_example(combined_data, st.session_state.selected_example_id)
        else:
            st.error("No combined data available")

    elif st.session_state.view_mode == "partitions":
        display_partition_overview(explorer)

    elif st.session_state.view_mode == "partition_clusters":
        display_partition_clusters(explorer, st.session_state.selected_partition)

    elif st.session_state.view_mode == "examples":
        current_step = st.session_state.navigation_path[-1]
        display_examples(explorer, current_step["cluster_id"])

    elif st.session_state.view_mode == "clusters":
        if st.session_state.navigation_path:
            # Show child clusters
            current_step = st.session_state.navigation_path[-1]
            current_level = current_step["level"]
            current_cluster_id = current_step["cluster_id"]

            children = explorer.get_child_clusters(current_level, current_cluster_id)

            if len(children) > 0:
                child_level = current_level - 1
                display_cluster_table(
                    children, child_level, st.session_state.color_assignments, explorer
                )
            else:
                st.warning("No child clusters found.")
                display_examples(explorer, current_cluster_id)
        else:
            # Show top level clusters
            top_level_data = explorer.data.get(f"level_{explorer.max_level}")
            if top_level_data is not None:
                display_cluster_table(
                    top_level_data,
                    explorer.max_level,
                    st.session_state.color_assignments,
                    explorer,
                )
            else:
                # Fallback to level 0 if no higher levels
                level_0_data = explorer.data.get("level_0")
                if level_0_data is not None:
                    display_cluster_table(
                        level_0_data, 0, st.session_state.color_assignments, explorer
                    )


if __name__ == "__main__":
    main()
