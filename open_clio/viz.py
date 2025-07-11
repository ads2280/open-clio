# evals - take csvs, make an experiment in langsmith, and then runs
import ast
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="OpenCLIO Clustering Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)


class ClusteringExplorer:
    def __init__(self, config_path: str = ".data/config_customer50.json"):  # TODO
        self.config = self.load_config(config_path)
        self.data = {}
        self.save_path = ".data/clustering_results"

        # find available levels withoutusing config
        self.available_levels = []
        self.max_level = 0
        self.level_names = []

    def load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        with open(config_path, "r") as f:
            return json.load(f)

    def detect_available_levels(self) -> List[int]:
        """Scan the results directory to detect which level files actually exist"""
        available_levels = []
        level_num = 0

        while True:
            level_path = os.path.join(self.save_path, f"level_{level_num}_clusters.csv")
            if os.path.exists(level_path):
                available_levels.append(level_num)
                level_num += 1
            else:
                break

        return available_levels

    @st.cache_data
    def load_data(_self) -> Dict[str, pd.DataFrame]:
        """Load combined.csv and dynamically detect available level data files"""
        data = {}
        save_path = _self.save_path

        # Load combined csv
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

        # Figure out how many levels exist and load level_{x}_cluster csvs
        _self.available_levels = _self.detect_available_levels()

        if not _self.available_levels:
            st.error("No level_X_clusters.csv files found!")
            st.stop()

        _self.max_level = max(_self.available_levels)
        _self.level_names = [f"level_{i}" for i in _self.available_levels]

        for level_num in _self.available_levels:
            level_name = f"level_{level_num}"
            level_path = os.path.join(save_path, f"{level_name}_clusters.csv")
            try:
                data[level_name] = pd.read_csv(level_path)
                st.success(f"Loaded {level_name} with {len(data[level_name])} clusters")
            except Exception as e:
                st.error(f"Error loading {level_name}_clusters.csv: {e}")
                st.stop()

        return data

    def parse_member_clusters(self, member_str: str) -> List[int]:
        """Parse member_clusters from string to list of ints"""
        if pd.isna(member_str) or member_str == "":
            return []
        elif "np.int32" in str(member_str):
            numbers = re.findall(r"np\.int32\((\d+)\)", str(member_str))
            return [int(num) for num in numbers]
        else:
            try:
                return ast.literal_eval(str(member_str))
            except:
                return []

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
        """Assign colors to all clusters across all available levels"""
        color_assignments = {}

        # Start with the highest available level
        top_level_key = f"level_{self.max_level}"
        if top_level_key not in self.data:
            # If no higher levels, just assign colors to level_0
            level_0_data = self.data.get("level_0")
            if level_0_data is not None:
                distinct_colors = self.get_distinct_colors(len(level_0_data))
                for i, (_, cluster_row) in enumerate(level_0_data.iterrows()):
                    cluster_id = cluster_row["cluster_id"]
                    color_assignments[(0, cluster_id)] = distinct_colors[
                        i % len(distinct_colors)
                    ]
            return color_assignments

        # Multi-level color assignment
        distinct_colors = self.get_distinct_colors(20)
        top_level_data = self.data[top_level_key]

        for i, (_, cluster_row) in enumerate(top_level_data.iterrows()):
            cluster_id = cluster_row["cluster_id"]
            base_color = distinct_colors[i % len(distinct_colors)]
            color_assignments[(self.max_level, cluster_id)] = base_color

            self._assign_child_colors(
                self.max_level, cluster_id, base_color, color_assignments
            )

        return color_assignments

    def _assign_child_colors(
        self, level: int, cluster_id: int, base_color: str, color_assignments: Dict
    ):
        """Recursively assign colors to child clusters"""
        if level == 0:
            return

        # Check if this level actually exists
        if level not in self.available_levels:
            return

        current_data = self.data.get(f"level_{level}")
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

        # Check if this level exists
        if level not in self.available_levels:
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

        # Check if child level exists
        child_level = level - 1
        if child_level not in self.available_levels:
            return pd.DataFrame()

        child_data = self.data.get(f"level_{child_level}")
        if child_data is None:
            return pd.DataFrame()

        return child_data[child_data["cluster_id"].isin(child_cluster_ids)].copy()


def build_hierarchy_display(explorer: ClusteringExplorer) -> str:
    """Build a hierarchy display string showing actual available levels"""
    if not explorer.available_levels:
        return "No levels available"

    hierarchy_parts = []
    for level in explorer.available_levels:
        level_data = explorer.data.get(f"level_{level}")
        if level_data is not None:
            count = len(level_data)
            if level == 0:
                hierarchy_parts.append(f"{count} base")
            elif level == explorer.max_level:
                hierarchy_parts.append(f"{count} top")
            else:
                hierarchy_parts.append(f"{count}")

    return " → ".join(hierarchy_parts) + " clusters"


def show_cluster_previews_in_sidebar(explorer: ClusteringExplorer):
    """Show cluster previews for each available level in sidebar"""
    st.divider()
    st.markdown("### Level Previews")

    for level in reversed(explorer.available_levels):  # Show from top to bottom
        level_data = explorer.data.get(f"level_{level}")
        if level_data is not None:
            with st.expander(
                f"Level {level} ({len(level_data)} clusters)",
                expanded=(level == explorer.max_level),
            ):
                # Show top 3 clusters by size
                top_clusters = level_data.nlargest(3, "size")
                for _, cluster in top_clusters.iterrows():
                    st.markdown(f"**{cluster['name']}** ({cluster['size']} items)")


def display_parent_stack():
    """Display navigation breadcrumb"""
    if "navigation_path" not in st.session_state:
        st.session_state.navigation_path = []

    if st.session_state.navigation_path:
        breadcrumbs = []
        for i, step in enumerate(st.session_state.navigation_path):
            # if st.button(f"Level {step['level']}: {step['name']}", key=f"breadcrumb_{i}"):
            # Go back to this level
            #    st.session_state.navigation_path = st.session_state.navigation_path[:i+1]
            #    st.session_state.view_mode = "clusters"
            #    st.rerun()
            breadcrumbs.append(
                f"Level {step['level']}: {step['name']}"
            )  # added name, can remove

        st.markdown(f"**Path:**\n" + "\n→ ".join(breadcrumbs))

        # if st.button("Home"):
        #    st.session_state.navigation_path = []
        #    st.session_state.view_mode = "clusters"
        #    st.rerun()
        # redundant now that back button added


def display_cluster_table(
    df: pd.DataFrame, level: int, color_assignments: Dict, explorer: ClusteringExplorer
):
    """Display a table of clusters with navigation"""
    st.markdown(f"### Level {level} Clusters ({len(df)} total)")

    for _, cluster_row in df.iterrows():
        cluster_id = cluster_row["cluster_id"]

        col1, col2 = st.columns([8, 2])

        with col1:
            # Show cluster name with partition badge if available
            # cluster_name = f"**{cluster_row['name']}**"
            # if "partition" in cluster_row and pd.notna(cluster_row["partition"]):
            #    partition = cluster_row["partition"]
            #    # Create a small badge-like display for the partition
            #    st.markdown(f"{cluster_name} 🏷️ **{partition}**")
            # else:
            #    st.markdown(cluster_name)
            st.markdown(f"**{cluster_row['name']}**")
            if "description" in cluster_row and pd.notna(cluster_row["description"]):
                st.markdown(f"{cluster_row['description']}")

        with col2:
            # Show partition information if available
            if (
                "partition" in cluster_row
                and pd.notna(cluster_row["partition"])
                and explorer.config["partitions"] is not None
            ):
                partition = cluster_row["partition"]
                st.markdown(f"**🏷️ {partition}**")

            # Check if this cluster has children
            if level > 0:
                st.metric("Sub-clusters", cluster_row.get("size", 0))
                children = explorer.get_child_clusters(level, cluster_id)
                if len(children) > 0:
                    if st.button(f"Explore →", key=f"explore_{level}_{cluster_id}"):
                        st.session_state.navigation_path.append(
                            {
                                "level": level,
                                "cluster_id": cluster_id,
                                "name": cluster_row["name"],
                            }
                        )
                        st.session_state.view_mode = "clusters"
                        st.rerun()
                else:
                    if st.button(f"Examples →", key=f"examples_{level}_{cluster_id}"):
                        st.session_state.navigation_path.append(
                            {
                                "level": level,
                                "cluster_id": cluster_id,
                                "name": cluster_row["name"],
                            }
                        )
                        st.session_state.view_mode = "examples"
                        st.session_state.selected_cluster_id = cluster_id
                        st.rerun()
            else:
                st.metric("Size", cluster_row.get("size", 0))
                if st.button(f"Examples →", key=f"examples_{level}_{cluster_id}"):
                    st.session_state.navigation_path.append(
                        {
                            "level": level,
                            "cluster_id": cluster_id,
                            "name": cluster_row["name"],
                        }
                    )
                    st.session_state.view_mode = "examples"
                    st.session_state.selected_cluster_id = cluster_id
                    st.rerun()

        st.divider()


def display_examples(explorer: ClusteringExplorer, cluster_id):
    """Display examples for a specific cluster"""
    combined_data = explorer.data.get("combined")
    if combined_data is None:
        st.error("No combined data available")
        return

    # Add back button at the top
    if st.button("← Back"):
        if st.session_state.navigation_path:
            st.session_state.navigation_path.pop()
        st.session_state.view_mode = "clusters"
        st.rerun()

    # Filter examples for this cluster at any level
    cluster_examples = combined_data[
        (combined_data["base_cluster_id"] == str(cluster_id))
        | (combined_data.get("top_cluster_id", "") == str(cluster_id))
    ]

    st.markdown(f"### Examples ({len(cluster_examples)} total)")

    for _, example in cluster_examples.iterrows():
        with st.expander(f"Example: {example['summary']}"):
            st.markdown(f"**Summary:** {example['summary']}")
            if "full_example" in example and pd.notna(example["full_example"]):
                st.markdown("**Full Example:**")
                st.text(example["full_example"])


def main():
    st.markdown("# Explore Clio Clusters")

    # Initialize session state
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "clusters"
    if "navigation_path" not in st.session_state:
        st.session_state.navigation_path = []
    if "color_assignments" not in st.session_state:
        st.session_state.color_assignments = {}

    try:
        explorer = ClusteringExplorer()
        explorer.data = explorer.load_data()
        explorer.available_levels = explorer.detect_available_levels()
        explorer.max_level = max(explorer.available_levels)
        explorer.level_names = [f"level_{i}" for i in explorer.available_levels]

        if not explorer.data:
            st.error("No data loaded. Please check your configuration and data files.")
            return

        # Assign colors once
        if not st.session_state.color_assignments:
            st.session_state.color_assignments = explorer.assign_colors_to_clusters()

    except Exception as e:
        st.error(f"Error initializing explorer: {e}")
        st.error("Please check your configuration and data files.")
        return

    # Show dataset info with actual hierarchy display
    hierarchy_display = build_hierarchy_display(explorer)
    st.markdown(f"**Dataset:** {explorer.config['dataset_name']}")
    st.markdown(f"**Hierarchy:** {hierarchy_display}")
    if explorer.config["partitions"] is not None:
        partitions_pretty_print = ", ".join(
            [f"{k}" for k in explorer.config["partitions"].keys()]
        )
    else:
        partitions_pretty_print = "None"
    st.markdown(f"**Partitions (🏷️):** {partitions_pretty_print}")

    # Nav breadcrumb
    display_parent_stack()

    # Shov examples or clusters based on what we're viewing currently
    if st.session_state.view_mode == "examples":
        if "selected_cluster_id" in st.session_state:
            display_examples(explorer, st.session_state.selected_cluster_id)
        elif st.session_state.navigation_path:
            current_step = st.session_state.navigation_path[-1]
            display_examples(explorer, current_step["cluster_id"])

    elif st.session_state.view_mode == "clusters":
        if st.session_state.navigation_path:
            if st.button("← Back"):
                st.session_state.navigation_path.pop()
                st.session_state.view_mode = "clusters"
                st.rerun()

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
                # or fallback to level 0
                level_0_data = explorer.data.get("level_0")
                if level_0_data is not None:
                    if explorer.max_level > 0:
                        if st.button(f"← Back to Level {explorer.max_level} Clusters"):
                            st.session_state.view_mode = "clusters"
                            st.rerun()
                    display_cluster_table(
                        level_0_data, 0, st.session_state.color_assignments, explorer
                    )


if __name__ == "__main__":
    main()
