import sys
import argparse
import asyncio
import json
import os
import streamlit.web.cli as stcli
from open_clio.generate import generate_clusters


def load_config(config_path=None):
    if config_path is None:
        config_path = "./config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    return config


def run_generate(config):
    print("Kicking off clustering pipeline...")
    print(f"Dataset: {config['dataset_name']}")
    print(f"Hierarchy (number of examples at each level): {config['hierarchy']}\n")
    print(f"Current working directory: {os.getcwd()}")

    asyncio.run(generate_clusters(**config))
    print("Clustering complete!")


def run_viz(config):
    print("Starting Clio visualization...")
    # Extract the specific values that viz needs and pass them as environment variables
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


def main():
    parser = argparse.ArgumentParser(
        description="OpenCLIO - Open-source implementation of CLIO clustering and visualization tool",
        epilog="""
examples:
  open-clio generate --config config.json    # generate clustering and launch visualization
  open-clio viz --config config.json         # launch visualization only
  open-clio evaluate --config config.json    # run evaluation on generated clusters

For more information, see the README or visit the project repository.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=["generate", "evaluate", "viz"],
        help="Action to perform (generate: clustering + viz, evaluate: run metrics, viz: visualization only)",
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
    elif args.action == "evaluate":
        run_evaluate(config)
    elif args.action == "viz":
        run_viz(config)
    else:
        print(f"Invalid action: {args.action}")
        exit(1)


if __name__ == "__main__":
    main()
