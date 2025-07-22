import sys
import argparse
import asyncio
import json
import os
import streamlit.web.cli as stcli
from datetime import datetime, timedelta


def load_config(config_path=None):
    if config_path is None:
        config_path = "./config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def process_time_config(config):
    # Validate only one time method is used
    has_explicit_times = bool(config.get("start_time") or config.get("end_time"))
    has_hours = config.get("hours") is not None
    has_days = config.get("days") is not None

    time_methods = sum([has_explicit_times, has_hours, has_days])
    if time_methods > 1:
        raise ValueError(
            "Only one of (start_time/end_time), hours, or days can be specified"
        )

    time_info = {"method": "explicit", "original": None}

    if has_explicit_times:
        # Convert string timestamps to datetime objects for validation
        start_time = config.get("start_time")
        end_time = config.get("end_time")

        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            config["start_time"] = start_time
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            config["end_time"] = end_time

        if start_time > end_time:
            raise ValueError("start_time must be before end_time")

    # Convert hours/days to start_time/end_time
    if has_hours:
        hours = config["hours"]
        if not isinstance(hours, int) or hours <= 0:
            raise ValueError("hours must be a positive integer")
        config["start_time"] = datetime.now() - timedelta(hours=hours)
        config["end_time"] = datetime.now()
        time_info = {"method": "hours", "original": hours}
        del config["hours"]

    elif has_days:
        days = config["days"]
        if not isinstance(days, int) or days <= 0:
            raise ValueError("days must be a positive integer")
        config["start_time"] = datetime.now() - timedelta(days=days)
        config["end_time"] = datetime.now()
        time_info = {"method": "days", "original": days}
        del config["days"]

    return config, time_info


def run_generate_langgraph(config):
    # validate config
    # sample
    if not config.get("sample"):
        config["sample"] = 2000

    # general
    if config.get("dataset_name") and config.get("project_name"):
        raise ValueError("dataset_name and project_name cannot both be provided")
    if not config.get("dataset_name") and not config.get("project_name"):
        raise ValueError("dataset_name or project_name must be provided")

    # time
    config, time_info = process_time_config(config)

    # dataset
    if config.get("dataset_name") and (
        config.get("start_time") or config.get("end_time")
    ):
        raise ValueError(
            "start_time and end_time cannot be provided when dataset_name is provided"
        )

    print("Starting Clio clustering pipeline...\n")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Dataset: {config['dataset_name']}") if config.get(
        "dataset_name"
    ) else print(f"Project: {config['project_name']}")
    print(f"Max sample size: {config['sample']}")
    if config.get("filter_string"):
        print(f"Filter string: {config['filter_string']}")

    from open_clio.generate_langgraph import run_graph, save_langgraph_results

    # Display time range information in a cleaner format
    if config.get("project_name"):
        start_time = config.get("start_time")
        end_time = config.get("end_time")

        if not start_time:
            start_time = datetime.now() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.now()

        # Format the time range display based on how it was specified
        if time_info["method"] == "hours":
            print(
                f"Time range: Last {time_info['original']} hours ({start_time.isoformat()} → {end_time.isoformat()})\n"
            )
        elif time_info["method"] == "days":
            print(
                f"Time range: Last {time_info['original']} days ({start_time.isoformat()} → {end_time.isoformat()})\n   "
            )
        else:
            print(f"Time range: {start_time.isoformat()} → {end_time.isoformat()}\n")

    # TODO add more checks (start_time > end_time, start_time > curr_time)

    dataset_name = config.get("dataset_name")
    project_name = config.get("project_name")
    start_time = config.get("start_time")
    end_time = config.get("end_time")
    hierarchy = config.get("hierarchy")
    summary_prompt = config.get("summary_prompt")
    save_path = config.get("save_path", "./clustering_results")
    partitions = config.get("partitions")
    sample = config.get("sample", 2000)
    max_concurrency = config.get("max_concurrency", 10)
    filter_string = config.get("filter_string")

    results = asyncio.run(
        run_graph(
            hierarchy,
            dataset_name=dataset_name,
            project_name=project_name,
            start_time=start_time,
            end_time=end_time,
            summary_prompt=summary_prompt,
            save_path=save_path,
            partitions=partitions,
            sample=sample,
            max_concurrency=max_concurrency,
            filter_string=filter_string,
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


# TODO - test it with projects, different timing configs
def run_extend(config):
    print("Starting cluster extension pipeline with LangGraph...")

    # Process time config first (convert hours/days to start_time/end_time)
    config, _ = process_time_config(config)

    save_path = config.get("save_path", "./clustering_results")
    dataset_name = config.get("dataset_name")
    project_name = config.get("project_name")
    sample = config.get("sample", None)

    # Validate that we have a dataset_name (extend currently only supports datasets)
    if not dataset_name:
        raise ValueError(
            "extend currently only supports datasets, not projects. Please provide dataset_name."
        )

    # Warn if time filtering is specified (since extend doesn't use it yet)
    if config.get("start_time") or config.get("end_time"):
        print(
            "Warning: Time filtering (start_time/end_time) is not currently used in extend mode."
        )

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

    if args.action == "generate":
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
