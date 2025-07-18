import sys
import argparse
import asyncio
import json
import os
import streamlit.web.cli as stcli
from datetime import datetime


def load_config(config_path=None):
    if config_path is None:
        config_path = "./config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def run_generate_langgraph(config):
    print("Starting Clio clustering pipeline with LangGraph...")
    print(f"Dataset: {config['dataset_name']}") if config.get(
        "dataset_name"
    ) else print(f"Project: {config['project_name']}")
    print(f"Hierarchy (number of examples at each level): {config['hierarchy']}\n")
    print(f"Current working directory: {os.getcwd()}")

    from open_clio.generate_langgraph import run_graph, save_langgraph_results

    # validate config
    # general
    if config.get("dataset_name") and config.get("project_name"):
        raise ValueError("dataset_name and project_name cannot both be provided")
    if not config.get("dataset_name") and not config.get("project_name"):
        raise ValueError("dataset_name or project_name must be provided")
    if not config.get("hierarchy"):
        raise ValueError("hierarchy must be provided")
    if not config.get("summary_prompt"):  # could add fallback
        raise ValueError(
            "summary_prompt must be provided, for example: Summarize this run: {{inputs.messages}}"
        )  # checkexample

    # dataset
    if config.get("dataset_name") and (
        config.get("start_time") or config.get("end_time")
    ):
        raise ValueError(
            "start_time and end_time cannot be provided when dataset_name is provided"
        )

    # project - edit if we change default start/end time
    if config.get("project_name") and not config.get("start_time"):
        print("Using start_time, datetime.now() - timedelta(hours=1)\n")
    if config.get("project_name") and not config.get("s_time"):
        print("Using default end_time, datetime.now()\n")

    # TODO add more checks (start_time > end_time, start_time > curr_time)

    dataset_name = config.get("dataset_name")
    project_name = config.get("project_name")
    start_time = config.get("start_time")
    end_time = config.get("end_time")
    hierarchy = config["hierarchy"]
    summary_prompt = config.get("summary_prompt")
    save_path = config.get("save_path", "./clustering_results")
    partitions = config.get("partitions")
    sample = config.get("sample")
    max_concurrency = config.get("max_concurrency", 10)

    results = asyncio.run(
        run_graph(
            dataset_name=dataset_name,
            project_name=project_name,
            start_time=start_time,
            end_time=end_time,
            hierarchy=hierarchy,
            summary_prompt=summary_prompt,
            save_path=save_path,
            partitions=partitions,
            sample=sample,
            max_concurrency=max_concurrency,
        )
    )

    save_langgraph_results(results, save_path)


def run_viz(config):
    print("Launching cluster visualization...")
    save_path = config.get("save_path", "./clustering_results")
    dataset_name = config.get("dataset_name")
    partitions = config.get("partitions")

    os.environ["CLIO_SAVE_PATH"] = save_path
    if dataset_name:
        os.environ["CLIO_DATASET_NAME"] = dataset_name
    if partitions:
        os.environ["CLIO_PARTITIONS"] = json.dumps(partitions)

    sys.argv = ["streamlit", "run", "open_clio/viz.py"]
    sys.exit(stcli.main())


def run_evaluate(config):
    from open_clio.evaluate import main as evaluate_main

    evaluate_main(config)


def run_extend(config):
    print("Starting cluster extension pipeline with LangGraph...")

    save_path = config.get("save_path", "./clustering_results")
    dataset_name = config.get("dataset_name")
    sample = config.get("sample", None)

    print(f"Loading existing hierarchy from: {save_path}")
    print(f"Loading new examples from dataset: {dataset_name}")
    print(f"Examples limit: {sample}")

    from open_clio.extend_langgraph import run_graph

    results = asyncio.run(run_graph(dataset_name, save_path, sample))

    print(
        f"Extension complete! Processed {results.get('processed_count', 0)} examples, skipped {results.get('skipped_count', 0)} previously clustered examples"
    )
    print(
        f"Run open-clio viz to explore the extended clusters, or see updated csv files in the {save_path} directory"
    )


def main():
    parser = argparse.ArgumentParser(
        description="OpenCLIO - Open-source implementation of CLIO clustering and visualization tool",
        epilog="""
examples:
  open-clio generate --config config.json    # generate clustering and launch visualization
  open-clio viz --config config.json         # launch visualization only
  open-clio evaluate --config config.json    # run evaluation on generated clusters
  open-clio extend --config config.json      # extend existing clusters with new examples

For more information, see the README or visit the project repository.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=["generate", "evaluate", "viz", "extend"],
        help="Action to perform (generate: clustering + viz, evaluate: run metrics, viz: visualization only, extend: add new examples to existing clusters)",
    )
    parser.add_argument(
        "--config",
        default="./config.json",
        help="Path to configuration file (default: ./config.json)",
    )

    args = parser.parse_args()

    if args.action is None:
        parser.print_help()
        sys.exit(1)

    config = load_config(args.config)

    if args.action == "generate":  # changed for langgraph
        run_generate_langgraph(config)
        run_viz(config)
    elif args.action == "evaluate":
        run_evaluate(config)
    elif args.action == "viz":
        run_viz(config)
    elif args.action == "extend":
        run_extend(config)
    else:
        print(f"Invalid action: {args.action}")
        exit(1)


if __name__ == "__main__":
    main()
