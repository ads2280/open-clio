# TODO update for threads/runs
import asyncio
from typing_extensions import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langsmith import schemas as ls_schemas
from langgraph.types import Send
from open_clio.extend import (
    load_examples,
    load_hierarchy,
    process_single_examples,
    extend_results,
)


class ExtendState(TypedDict):
    # configs
    dataset_name: str
    save_path: str
    sample: int | None

    # existing hierarchy
    existing_data: dict  # combined.csv + level_x_clusters.csv
    existing_eids: set[str]

    # new examples to add to hierarchy
    new_examples: Annotated[list[ls_schemas.Example], lambda l, r: l + r]

    # results
    assignments: Annotated[list[dict], lambda l, r: l + r]
    skipped_count: Annotated[int, lambda l, r: l + r]
    processed_count: Annotated[int, lambda l, r: l + r]


def load_hierarchy_node(state: ExtendState) -> dict:
    save_path = state["save_path"]
    existing_data = load_hierarchy(save_path)
    existing_eids = set(existing_data["combined_df"]["example_id"].tolist())
    return {"existing_data": existing_data, "existing_eids": existing_eids}


def load_examples_node(state: ExtendState) -> dict:
    dataset_name = state["dataset_name"]
    sample = state["sample"]
    existing_eids = state["existing_eids"]

    examples = load_examples(dataset_name, sample)
    new_examples = [e for e in examples if str(e.id) not in existing_eids]

    skipped_count = len(examples) - len(new_examples)
    processed_count = len(new_examples)

    return {
        "new_examples": new_examples,
        "skipped_count": skipped_count,
        "processed_count": processed_count,
    }


# def filter_new_examples(state: ExtendState) -> dict:
# don't need anymore bc its in load_example


# main part to parallelize
def map_assign_examples(state: ExtendState) -> list[Send]:
    return [
        Send(
            "assign_single_example",
            {
                "example": example,
                "existing_data": state["existing_data"],
            },
        )
        for example in state["new_examples"]
    ]


async def assign_single_example(
    state: ExtendState,
) -> dict:
    example = state["example"]
    existing_data = state["existing_data"]
    # where the magic happens
    # although prompts pretty basic currently
    assignment = await process_single_examples(example, existing_data)
    return {"assignments": [assignment]}


def aggregate_assignments(state: ExtendState) -> dict:
    return {}


def extend_results_node(state: ExtendState) -> dict:
    assignments = state["assignments"]
    save_path = state["save_path"]
    extend_results(assignments, save_path)
    return {}


cluster_extender = StateGraph(ExtendState)
cluster_extender.add_node(load_hierarchy_node)
cluster_extender.add_node(load_examples_node)
cluster_extender.add_node(assign_single_example)  # parallel
cluster_extender.add_node(aggregate_assignments)
cluster_extender.add_node(extend_results_node)

cluster_extender.set_entry_point("load_hierarchy_node")
cluster_extender.add_edge("load_hierarchy_node", "load_examples_node")
cluster_extender.add_conditional_edges(
    "load_examples_node", map_assign_examples, ["assign_single_example"]
)
cluster_extender.add_edge("assign_single_example", "aggregate_assignments")
cluster_extender.add_edge("aggregate_assignments", "extend_results_node")
cluster_extender.add_edge("extend_results_node", END)

cluster_extender_graph = cluster_extender.compile()


async def run_graph(dataset_name: str, save_path: str, sample: int | None = None):
    results = await cluster_extender_graph.ainvoke(
        {
            "dataset_name": dataset_name,
            "save_path": save_path,
            "sample": sample,
        },
        config={
            "recursion_limit": 100,
        },
    )
    return results
