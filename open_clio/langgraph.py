import warnings
from collections import defaultdict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.types import Send
from open_clio.generate import (
    save_results,
    DEFAULT_SUMMARIZATION_CONCURRENCY,
    validate_hierarchy,
    summarize_example,
)
from logging import getLogger
from langsmith import Client
from langsmith import schemas as ls_schemas
from open_clio.internal.schemas import ExampleSummary

DEFAULT_SUMMARY_PROMPT = """summarize por favor: {{example}}"""

logger = getLogger(__name__)


class State(TypedDict):
    dataset_name: str
    sample: int | float | None
    hierarchy: list
    partitions: dict | None
    summary_prompt: str | None
    examples: list[ls_schemas.Example]
    summaries: list[ExampleSummary]
    total_examples: int
    clusters: dict


class Config(TypedDict):
    max_concurrency: int | None


def load_examples(state: State) -> dict:
    partitions = state["partitions"]
    hierarchy = state["hierarchy"]
    dataset_name = state["dataset_name"]
    if partitions is not None:
        num_partitions = len(partitions.keys())
        num_top_level_clusters = hierarchy[-1]
        if num_partitions != num_top_level_clusters:
            warnings.warn(
                f"Number of partitions ({num_partitions}) does not match number of "
                f"top-level clusters ({num_top_level_clusters})"
            )

    # load data
    logger.info(f"Loading and summarizing examples from '{dataset_name}' dataset")
    print(f"Loading dataset '{dataset_name}'...")

    client = Client()
    examples = list(
        client.list_examples(
            dataset_name=dataset_name,
            limit=state["sample"] if state.get("sample") else None,
        )
    )
    total_examples = len(examples)
    validate_hierarchy(hierarchy, total_examples)  # Gives you an option to quit

    logger.info(f"Loaded {total_examples} total examples, generating summaries...")
    print(f"Loaded {total_examples} examples, generating summaries...")
    return {"total_examples": total_examples, "examples": examples}


async def summarize(state: State) -> dict:
    example = state["example"]
    summary = await summarize_example(
        example,
        state["partitions"],
        state.get("summary_prompt", DEFAULT_SUMMARY_PROMPT),
    )
    return {"summaries": [summary]}


def map_summaries(state: State) -> list[Send]:
    return [Send("summarize", {"example": e}) for e in state["examples"]]


def map_partitions(state: State) -> list[Send]:
    summaries_by_partition = defaultdict(list)
    # Prepare to process examples by partition
    for summary in state["summaries"]:
        if summary:
            summaries_by_partition[summary["partition"]].append(summary)

    logger.info(
        f"The dataset contains the following partitions: {list(summaries_by_partition)}"
    )
    print(f"Partitions: {list(summaries_by_partition.keys())}")

    # Process partitions one at a time and append to an updates list
    all_updates = []
    combined_hierarchy = {"partitions": {}}

    sends = []
    for partition, cat_summaries in summaries_by_partition.items():
        example_ids = [s["example_id"] for s in cat_summaries]
        partition_examples = [e for e in state["examples"] if e.id in example_ids]
        sends.append(
            Send(
                "partition",
                {
                    "partition": partition,
                    "partition_examples": partition_examples,
                    "cat_summaries": cat_summaries,
                },
            )
        )
    return sends


async def partition(state: State) -> dict:
    logger.info(f"Clustering examples that belong to partition '{partition}'")
    print(f"\nProcessing partition '{partition}'...")

    try:
        partition_updates, partition_hierarchy = cluster_partition_examples(
            state["partition"],
            state["partition_examples"],
            state["cat_summaries"],
            state["total_examples"],
            state["hierarchy"],
        )
    except Exception as e:
        logger.error(f"ERROR processing partition {partition}: {e}")
        print(f"ERROR processing partition {partition}: {e}")
    else:
        all_updates.extend(partition_updates)
        combined_hierarchy["partitions"][partition] = partition_hierarchy

        logger.info("Searching for more partitions to cluster...")
        time.sleep(1.0)


graph_builder = StateGraph(State)
graph_builder.add_node(load_examples)
graph_builder.add_node(summarize)

graph_builder.set_entry_point("load_examples")
graph_builder.add_conditional_edges("load_examples", map_summaries, ["summarize"])
graph_builder.add_conditional_edges("summarize", map_partitions, ["partition"])
graph = graph_builder.compile()


async def run_graph(
    dataset_name: str,
    hierarchy: list,
    summary_prompt: str,  # TODO
    *,
    save_path: str | None = None,
    partitions: dict | None = None,
    sample: int | None = None,
    max_concurrency: int = DEFAULT_SUMMARIZATION_CONCURRENCY,
):
    results = await graph.ainvoke(
        {
            "dataset_name": dataset_name,
            "hierarchy": hierarchy,
            "partitions": partitions,
            "sample": sample,
        },
        config={"summary_prompt": summary_prompt, "max_concurrency": max_concurrency},
    )
    save_results(..., save_path)
