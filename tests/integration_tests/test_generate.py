from open_clio import generate_clusters

TEST_CONFIG = {
    "dataset_name": "ds-granular-pseudoscience-68",
    "hierarchy": [5],
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


def test_generate_clusters() -> None:
    clusters = generate_clusters(**TEST_CONFIG)
    assert clusters == ...