import streamlit as st
import pandas as pd

# Set page title
st.title("Hierarchical Data Visualization")

# File upload or selection
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Initialize session state for tracking selected row
if "selected_ids" not in st.session_state:
    st.session_state.selected_ids = []
    st.session_state.selected_names = []
if "current_level" not in st.session_state:
    st.session_state.current_level = None


# Function to load data
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    # Ensure all column types are correct
    df["id"] = df["id"].astype(int)
    df["level"] = df["level"].astype(int)
    # Convert parent_id to int where not null, keep as null otherwise
    df["parent_id"] = pd.to_numeric(df["parent_id"], errors="coerce")
    return df


# Function to filter data based on selection
def filter_data(df, parent_id=None, level=None):
    if parent_id is None:
        # Get rows with the highest level
        max_level = df["level"].max()
        return df[df["level"] == max_level]
    else:
        # Get rows with parent_id matching the selected row
        return df[(df["parent_id"] == parent_id) & (df["level"] == level)]


# Function to display navigation breadcrumbs
def display_breadcrumbs():
    if st.session_state.selected_ids:
        cols = st.columns(len(st.session_state.selected_ids) + 1)

        # Home button
        if cols[0].button("🏠 Home"):
            st.session_state.selected_ids = []
            st.session_state.selected_names = []
            st.rerun()

        # Path buttons
        for i, (id, name) in enumerate(
            zip(st.session_state.selected_ids, st.session_state.selected_names)
        ):
            display_name = name if name else f"ID: {id}"
            if cols[i + 1].button(f"📁 {display_name}", key=f"path_{i}"):
                # Go back to this level
                st.session_state.selected_ids = st.session_state.selected_ids[: i + 1]
                st.session_state.selected_names = st.session_state.selected_names[
                    : i + 1
                ]
                st.rerun()


# Main app logic
if uploaded_file is not None:
    # Load data
    df = load_data(uploaded_file)

    # Display navigation breadcrumbs
    display_breadcrumbs()

    # Get current parent ID (last selected ID or None)
    current_parent_id = (
        st.session_state.selected_ids[-1] if st.session_state.selected_ids else None
    )
    current_level = st.session_state.current_level

    # Filter data based on current selection
    filtered_df = filter_data(df, current_parent_id, current_level)

    current_level = current_level or filtered_df["level"].max()

    # Display header based on current view
    if current_parent_id is None:
        st.header(f"L={current_level}: Top Level Items")
    else:
        parent_name = (
            st.session_state.selected_names[-1]
            if st.session_state.selected_names[-1]
            else f"ID: {current_parent_id}"
        )
        st.header(f"L={current_level}: Children of '{parent_name}'")

    # Display the filtered data
    if len(filtered_df) > 0:
        # Create a simplified view with selected columns and better order
        display_df = filtered_df[["id", "name", "summary"]].copy()

        # Use dataframe with column config for better control
        st.dataframe(
            display_df,
            use_container_width=False,  # Don't constrain to container width
            width=2000,  # Set a fixed width (px) - adjust as needed
            height=min(6, len(display_df)) * 150,  # Set a reasonable height
            row_height=150,
            column_config={
                "id": st.column_config.Column(
                    "ID",
                    width="small",
                ),
                "name": st.column_config.Column(
                    "Name",
                    width="medium",
                ),
                "summary": st.column_config.TextColumn(
                    "Summary", width="large", help="Summary text"
                ),
            },
        )

        # Add spacing
        st.markdown("<br>", unsafe_allow_html=True)

        if current_level >= 1:
            # Add navigation section below table
            st.subheader(f"Navigate to children (L={current_level - 1}):")

            # Create columns for navigation buttons with more space
            num_cols = 2  # Reducing from 3 to 2 for more space
            cols = st.columns(num_cols)

            # Add buttons for each row that has children
            for i, (index, row) in enumerate(filtered_df.iterrows()):
                # Check if this row has children
                has_children = any(df["parent_id"] == row["id"])

                if has_children:
                    row_id = int(row["id"])
                    row_name = row["name"] if pd.notna(row["name"]) else f"ID: {row_id}"
                    col_idx = i % num_cols

                    if cols[col_idx].button(
                        f"ID {row_id}: {row_name}", key=f"row_{row_id}"
                    ):
                        # Add this row to the selected path
                        st.session_state.selected_ids.append(row_id)
                        st.session_state.selected_names.append(
                            row["name"] if pd.notna(row["name"]) else None
                        )
                        st.session_state.current_level = current_level - 1
                        st.rerun()
    else:
        st.info("No data found for the current selection.")
else:
    st.info("Please upload a CSV file to begin.")

# Add sidebar with info
with st.sidebar:
    st.header("About")
    st.write("""
    This app visualizes hierarchical data from a CSV file.
    
    The CSV should have these columns:
    - id: int
    - parent_id: int or None
    - level: int
    - name: str or None
    - summary: str or None
    
    Start by uploading a CSV file using the uploader above.
    """)
