from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langsmith import schemas as ls_schemas
from open_clio.internal.schemas import ExampleSummary
from langgraph.types import Send

class ExtendState(TypedDict):
    dataset_name: str
    save_path: str
    sample: int | None

    # existing hierarchy
    existing_data: dict  # combined.csv + level_x_clusters.csv
    existing_example_ids: set[str]

    # new examples to add to hierarchy
    new_examples: Annotated[list[ls_schemas.Example], lambda l, r: l + r]

    # results
    assignments: Annotated[list[dict], lambda l, r: l + r]  # Final assignments
    skipped_count: Annotated[int, lambda l, r: l + r]
    processed_count: Annotated[int, lambda l, r: l + r]

def load_existing_clusters(state: ExtendState) -> dict:
    pass

def load_new_examples(state: ExtendState) -> dict:
    pass

def filter_new_examples(state: ExtendState) -> dict:
    pass # figure out what's already in clusters and what's now


def map_process_examples(state: ExtendState) -> list[Send]:
    return [
        Send(
            "process_single_example",
            {
                "example": example,
                "existing_data": state["existing_data"],
            },
        )
        for example in state["new_examples"]
    ]

def process_single_example(state: ExtendState) -> dict:
    pass #extend.py's process_single_examples
    # 1. determine_partition_and_summarize()
    # 2. assign_to_base_cluster()
    # 3. assign_higher_levels()
    # return dict w assignment result

def aggregate_assignments(state: ExtendState) -> dict:
    pass

def save_results(state: ExtendState) -> dict:
    pass

cluster_extender = StateGraph(ExtendState)
cluster_extender.add_node(load_existing_clusters) 
cluster_extender.add_node(load_new_examples)
cluster_extender.add_node(filter_new_examples)
cluster_extender.add_node(process_single_example) #parallel
cluster_extender.add_node(aggregate_assignments)
cluster_extender.add_node(save_results)

cluster_extender.set_entry_point("load_existing_clusters")
cluster_extender.add_edge("load_existing_clusters", "load_new_examples")
cluster_extender.add_edge("load_new_examples", "filter_new_examples")
cluster_extender.add_edge("filter_new_examples", "process_single_example")
cluster_extender.add_edge("process_single_example", "aggregate_assignments")
cluster_extender.add_edge("aggregate_assignments", "save_results")
cluster_extender.add_edge("save_results", END)

cluster_extender.compile()
