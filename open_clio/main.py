import sys
import argparse
import asyncio
import json
import os
import streamlit.web.cli as stcli
from open_clio.generate import generate_clusters
from open_clio.extend import main as extend_main


def load_config(config_path=None):
    if config_path is None:
        config_path = "./config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def run_generate(config):
    print("Starting Clio clustering pipeline...")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Hierarchy (number of examples at each level): {config['hierarchy']}\n")
    print(f"Current working directory: {os.getcwd()}")

    asyncio.run(generate_clusters(**config))
    print("Clustering complete!")


def run_generate_langgraph(config):
    print("Starting Clio clustering pipeline with LangGraph...")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Hierarchy (number of examples at each level): {config['hierarchy']}\n")
    print(f"Current working directory: {os.getcwd()}")

    from open_clio.generate_langgraph import run_graph, save_langgraph_results

    dataset_name = config["dataset_name"]
    hierarchy = config["hierarchy"]
    summary_prompt = config.get("summary_prompt", "summarize por favor: {{example}}")
    save_path = config.get("save_path", "./clustering_results")
    partitions = config.get("partitions")
    sample = config.get("sample")
    max_concurrency = config.get("max_concurrency", 10)

    results = asyncio.run(
        run_graph(
            dataset_name=dataset_name,
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
    print("Starting cluster extension pipeline...")

    save_path = config.get("save_path", "./clustering_results")
    dataset_name = config.get("dataset_name")
    sample = config.get("sample", None)

    print(f"Loading existing hierarchy from: {save_path}")
    print(f"Loading new examples from dataset: {dataset_name}")
    print(f"Examples limit: {sample}")

    asyncio.run(extend_main(dataset_name, save_path, sample))

    print(
        f"\n\nExtension complete! Run open-clio viz to explore the extended clusters, or see updated csv files in the {save_path} directory"
    )


def main():
    parser = argparse.ArgumentParser(
        description="OpenCLIO - Open-source implementation of CLIO clustering and visualization tool",
        epilog="""
examples:
  open-clio generate --config config.json    # generate clustering and launch visualization
  open-clio langgraph --config config.json   # generate clustering with LangGraph and launch visualization
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
        choices=["generate", "langgraph", "evaluate", "viz", "extend"],
        help="Action to perform (generate: clustering + viz, langgraph: langgraph clustering + viz, evaluate: run metrics, viz: visualization only, extend: add new examples to existing clusters)",
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

    if args.action == "generate":
        run_generate(config)
        run_viz(config)
    elif args.action == "langgraph":
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
