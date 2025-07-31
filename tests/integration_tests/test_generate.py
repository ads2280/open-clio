from open_clio import generate_clusters
from pathlib import Path
import pandas as pd

TEST_CONFIG = {
    "dataset_name": "ds-timely-graduate-33",
    "hierarchy": [12],
    "summary_prompt": "Your job is to analyze this conversation and extract the key details about what the user is asking the AI assistant to do.\nFocus on capturing the main task, request, or purpose of the conversation in a clear, concise way.\n\nProvide a structured summary in this format:\n'[ACTION/TASK] with [SPECIFIC_TOPIC/SUBJECT] for [CONTEXT/PURPOSE]'\n\nExamples:\n- 'help with writing Python code for data analysis project'\n- 'explain machine learning concepts for academic research'\n- 'create marketing content for social media campaign'\n- 'debug software issues for web application development'\n- 'provide advice on career planning for recent graduate'\n- 'analyze financial data for investment decision making'\n- 'generate creative content for storytelling project'\n- 'answer questions about historical events for educational purposes'\n\nGuidelines:\n- Focus on what the user is asking the AI to do or help with\n- Be specific about the subject matter or domain when clear\n- Leave out redundant words like 'User requested' or 'I understand'\n- Include context about the purpose, use case, or technical details when relevant to the domain\n- Capture the core intent of the conversation\n- Don't include any personally identifiable information (PII) like names, locations, phone numbers, email addresses\n- Don't include any proper nouns\n- Be clear, descriptive and specific\n- Keep it concise - aim for under 20 words",
    "sample": None,
    "partitions": {
        "Admin/Account management": "Billing, Authentication, Account Management, Data Deletion, Security/Privacy",
        "LangChain OSS": "Python library, JavaScript library, LangMem and other components",
        "LangSmith product": "Evaluation, Dashboards, Annotations, Datasets & Experiments, Playground, SDK/Tracing, Prompt Hub",
        "LangGraph": "Platform (deployment/infra), OSS Python, OSS JavaScript, Studio, Pricing",
        "LangSmith deployment": "Setting up SSO, provisioning cloud resources, managing databases, helm/kubernetes/docker/AWS/GCP/Azure",
        "Other": "Sales inquiries, Spam, Unrelated issues",
    },
}

TEST_CONFIG_WITHOUT_PARTITIONS = {
    "dataset_name": "ds-timely-graduate-33",
    "hierarchy": [12, 3],
    "summary_prompt": "Your job is to analyze this conversation and extract the key details about what the user is asking the AI assistant to do.\nFocus on capturing the main task, request, or purpose of the conversation in a clear, concise way.\n\nProvide a structured summary in this format:\n'[ACTION/TASK] with [SPECIFIC_TOPIC/SUBJECT] for [CONTEXT/PURPOSE]'\n\nExamples:\n- 'help with writing Python code for data analysis project'\n- 'explain machine learning concepts for academic research'\n- 'create marketing content for social media campaign'\n- 'debug software issues for web application development'\n- 'provide advice on career planning for recent graduate'\n- 'analyze financial data for investment decision making'\n- 'generate creative content for storytelling project'\n- 'answer questions about historical events for educational purposes'\n\nGuidelines:\n- Focus on what the user is asking the AI to do or help with\n- Be specific about the subject matter or domain when clear\n- Leave out redundant words like 'User requested' or 'I understand'\n- Include context about the purpose, use case, or technical details when relevant to the domain\n- Capture the core intent of the conversation\n- Don't include any personally identifiable information (PII) like names, locations, phone numbers, email addresses\n- Don't include any proper nouns\n- Be clear, descriptive and specific\n- Keep it concise - aim for under 20 words",
    "sample": None,
    "partitions": None,
}


async def test_generate_clusters_with_partitions() -> None:
    result = await generate_clusters(**TEST_CONFIG)
    assert result is None, "generate_clusters should return None"

    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / ".data/clustering_results"
    assert results_dir.exists(), "results directory should exist"

    csv_files = ["combined.csv", "level_0_clusters.csv"]  # , "level_1_clusters.csv"]
    for filename in csv_files:
        assert (results_dir / filename).exists(), (
            f"{filename} should exist within results directory"
        )

    combined_df = pd.read_csv(results_dir / "combined.csv")
    level_0_df = pd.read_csv(results_dir / "level_0_clusters.csv")
    # level_1_df = pd.read_csv(results_dir / "level_1_clusters.csv")

    required_columns_combined = [
        "example_id",
        "full_example",
        "summary",
        "category",
        "base_cluster_id",
        "base_cluster_name",
        "top_cluster_id",
        "top_cluster_name",
    ]
    required_columns_level = ["cluster_id", "name", "description", "size", "category"]

    for col in required_columns_combined:
        assert col in combined_df.columns, f"combined.csv should have the column {col}"

    for col in required_columns_level:
        assert col in level_0_df.columns, (
            f"level_0_clusters.csv should have the column {col}"
        )

    # for col in required_columns_level:
    #    assert col in level_1_df.columns, f"level_1_clusters.csv should have the column {col}"

    assert len(level_0_df) > 0, "level_0_clusters.csv should have rows"
    assert len(level_0_df) < 19, "level_0_clusters.csv should have less than 19 rows"

    # assert len(level_1_df) > 0, "level_1_clusters.csv should have rows"
    # assert len(level_1_df) < 12, "level_1_clusters.csv should have less than 12 rows"

    assert len(combined_df) == 50, "combined.csv should have 50 rows"
    assert level_0_df["size"].sum() == 50, (
        "adding what's in each base cluster should match the number of examples"
    )
    # assert level_1_df["total_size"].sum() == 50, "adding what's in each intermediate cluster should match the number of examples"


async def test_generate_clusters_without_partitions() -> None:
    result = await generate_clusters(**TEST_CONFIG_WITHOUT_PARTITIONS)
    assert result is None, "generate_clusters should return None"

    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / ".data/clustering_results"
    assert results_dir.exists(), "results directory should exist"

    csv_files = ["combined.csv", "level_0_clusters.csv", "level_1_clusters.csv"]
    for filename in csv_files:
        assert (results_dir / filename).exists(), (
            f"{filename} should exist within results directory"
        )

    combined_df = pd.read_csv(results_dir / "combined.csv")
    level_0_df = pd.read_csv(results_dir / "level_0_clusters.csv")
    level_1_df = pd.read_csv(results_dir / "level_1_clusters.csv")

    required_columns_combined = [
        "example_id",
        "full_example",
        "summary",
        "category",
        "base_cluster_id",
        "base_cluster_name",
        "top_cluster_id",
        "top_cluster_name",
    ]
    required_columns_level = ["cluster_id", "name", "description", "size", "category"]

    for col in required_columns_combined:
        assert col in combined_df.columns, f"combined.csv should have the column {col}"

    for col in required_columns_level:
        assert col in level_0_df.columns, (
            f"level_0_clusters.csv should have the column {col}"
        )

    for col in required_columns_level:
        assert col in level_1_df.columns, (
            f"level_1_clusters.csv should have the column {col}"
        )

    assert len(level_0_df) > 0, "level_0_clusters.csv should have rows"
    assert len(level_0_df) < 14, "level_0_clusters.csv should have less than 14 rows"

    assert len(level_1_df) > 0, "level_1_clusters.csv should have rows"
    assert len(level_1_df) < 5, "level_1_clusters.csv should have less than 5 rows"

    assert len(combined_df) == 50, "combined.csv should have 50 rows"
    assert level_0_df["size"].sum() == 50, (
        "adding what's in each base cluster should match the number of examples"
    )
    assert level_1_df["total_size"].sum() == 50, (
        "adding what's in each intermediate cluster should match the number of examples"
    )

    # could do id matches
