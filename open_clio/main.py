import sys
import argparse
import asyncio
import json
import os
import streamlit.web.cli as stcli
from datetime import datetime, timedelta
from open_clio.generate import generate_clusters
from open_clio.extend import main as extend_main


def parse_datetime_string(datetime_str):
    """Parse datetime strings from config, supporting expressions like 'datetime.now() - timedelta(days=1)'"""
    if not datetime_str:
        return None
    
    # Handle common datetime expressions
    if datetime_str == "datetime.now()":
        return datetime.now()
    elif "datetime.now() - timedelta(days=" in datetime_str:
        # Extract days from string like "datetime.now() - timedelta(days=1)"
        days_str = datetime_str.split("timedelta(days=")[1].split(")")[0]
        days = int(days_str)
        return datetime.now() - timedelta(days=days)
    elif "datetime.now() - timedelta(hours=" in datetime_str:
        # Extract hours from string like "datetime.now() - timedelta(hours=1)"
        hours_str = datetime_str.split("timedelta(hours=")[1].split(")")[0]
        hours = int(hours_str)
        return datetime.now() - timedelta(hours=hours)
    else:
        # Try to parse as ISO format datetime string
        try:
            return datetime.fromisoformat(datetime_str)
        except ValueError:
            raise ValueError(f"Unsupported datetime format: {datetime_str}")


def load_config(config_path=None):
    if config_path is None:
        config_path = "./config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Parse datetime strings if present
    if "start_time" in config:
        config["start_time"] = parse_datetime_string(config["start_time"])
    if "end_time" in config:
        config["end_time"] = parse_datetime_string(config["end_time"])

    return config


def run_generate(config):
    print("Starting Clio clustering pipeline...")
    if config.get('dataset_name'):
        print(f"Dataset: {config['dataset_name']}")
    elif config.get('project_name'):
        print(f"Project: {config['project_name']}")
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


def run_extend(config):
    print("Starting cluster extension pipeline...")
    
    save_path = config.get("save_path", "./clustering_results")
    dataset_name = config.get("dataset_name")
    sample = config.get("sample", None)  
    
    print(f"Loading existing hierarchy from: {save_path}")
    print(f"Loading new examples from dataset: {dataset_name}")
    print(f"Examples limit: {sample}")
    
    asyncio.run(extend_main(dataset_name, save_path, sample))
    
    print(f"\n\nExtension complete! Run open-clio viz to explore the extended clusters, or see updated csv files in the {save_path} directory")


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

    if args.action == "generate":
        run_generate(config)
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
