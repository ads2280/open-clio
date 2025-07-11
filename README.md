# OpenCLIO

CLIO is a tool that extracts high level insights and patterns from datasets. It automatically summarizes each example in a dataset, groups similar examples together, and organizes them into a navigable hierarchy of descriptive clusters.

You provide a dataset, target hierarchy, summary prompt, and optionally define a set of partitions for organizing your data upfront. Clio summarizes each data point using your summary prompt to focus on what's most relevant, embeds and clusters those summaries based on semantic similarity, and iteratively groups clusters into higher-level categories. It also includes utilities that allow you to navigate through your clusters in an interactive web interface and evaluate how well the clustering captures the underlying patterns in your data.

For setup and usage details, see the quick start guide below.

## Quick Start

### 1. Install 
```bash
uv pip install -e .
```
Also make sure you have valid LangSmith and Anthropic API keys in your environment.

### 2. Set up your configuration
Choose a dataset to run Clio clustering on and create a `config.json` file, or skip this step and test with `example_config.json`.
```json
{
    "dataset_name": "your-langsmith-dataset-name",
    "hierarchy": [15, 8, 3],
    "summary_prompt": "Summarize what the user is asking the AI to help with. Focus on the main task or request.",
    "save_path": "./clustering_results",
}
```

**What these settings mean:**
- `dataset_name`: Your LangSmith dataset name
- `hierarchy`: How many clusters to create at each level and how many levels total [base_level, middle_level, top_level] (3 levels in this example)
- `summary_prompt`: Instructions specifying what domain-specific info the LLM should focus on when summarizing conversations 
- `save_path`: Where to save the results

### 3. Generate Clio clusters
```bash
uv run open-clio generate --config path/to/config
```

This will:
- Download your dataset from LangSmith
- Summarize each conversation
- Create clusters at multiple levels
- Launch a web interface to explore the results

### 4. Explore your results
When clustering is complete, the command above automatically opens a web browser where you can navigate through your clusters, from high-level categories down to individual examples.

You can also explore the output files in `save_path` for a quick overview of clusters:
- `combined.csv`: All conversations with their cluster assignments
- `level_0_clusters.csv`: Detailed cluster information
- `level_1_clusters.csv`: Medium-level cluster information
- `level_2_clusters.csv`: Broad category information


## Example Walkthrough

This example is pre-configured for a dataset of customer support conversations:

1. **Run**: `open-clio generate --config example_config.json` 
2. **Explore**: The web interface shows clusters like:
   - "Collect financial invoices and receipts from business organizations for accounting record management"
   - "Manage enterprise SaaS platform operations"
   - "Establish security frameworks and vulnerability management systems for open-source AI development projects"

   
## How to:
### Define partitions upfront
in progress

### Write a summary prompt
in progress

### Select a hierarchy
in progress

### Evaluate clusters
in progress

## Reference

### CLI arguments
```
usage: open-clio [-h] [--config CONFIG] [{generate,evaluate,viz}]

OpenCLIO - Open-source implementation of CLIO clustering and visualization tool

positional arguments:
  {generate,evaluate,viz}
                        Action to perform (generate: clustering + viz, evaluate: run metrics, viz: visualization only)

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to configuration file (default: ./config.json)

examples:
  open-clio generate --config config.json    # generate clustering and launch visualization
  open-clio viz --config config.json         # launch visualization only
  open-clio evaluate --config config.json    # run evaluation on generated clusters

For more information, see the README or visit the project repository.
```

### Configuration 
Your `config.json` should include the following arguments:
#### Required
- `dataset_name`: Your LangSmith dataset name
- `hierarchy`: List of cluster counts for each level
- `summary_prompt`: How to summarize conversations

#### Optional
- `save_path`: Where to save results (default: `./clustering_results`)
- `sample`: Process only N examples (default: all)
- `partitions`: Pre-defined categories for your data

This is an example config.json file:
```
{
"dataset_name": "unthread-data",
"hierarchy": [125,15],
"summary_prompt": "Your job is to analyze this customer service conversation and extract the key details about what the customer is asking for help with.\nFocus on capturing the main issue, request, or problem the customer is experiencing in a clear, concise way.\n\nProvide a structured summary in this format:\n'[ACTION/TASK] with [SPECIFIC_TOPIC/SUBJECT] for [CONTEXT/PURPOSE]'\n\nExamples:\n- 'resolve billing payment issues for subscription account'\n- 'troubleshoot login authentication problems for web application'\n- 'explain product feature functionality for new user onboarding'\n- 'process refund request for defective merchandise'\n- 'assist with account security settings for privacy concerns'\n- 'help with order tracking and delivery status for e-commerce'\n- 'provide technical support for software installation issues'\n- 'handle subscription cancellation request for billing department'\n\nGuidelines:\n- Focus on what the customer is asking for help with\n- Be specific about the product, service, or issue when clear\n- Leave out redundant words like 'Customer requested' or 'I understand'\n- Include context about the urgency, complexity, or technical details when relevant\n- Capture the core customer need or problem\n- Don't include any personally identifiable information (PII) like names, locations, phone numbers, email addresses\n- Don't include any proper nouns\n- Be clear, descriptive and specific\n- Keep it concise - aim for one clear sentence",
"sample": null,
"partitions": {
    "Admin/Account management": "Billing, Authentication, Account Management, Data Deletion, Security/Privacy",
    "LangChain OSS": "Python library, JavaScript library, LangMem and other components",
    "LangSmith product": "Evaluation, Dashboards, Annotations, Datasets & Experiments, Playground, SDK/Tracing, Prompt Hub",
    "LangGraph": "Platform (deployment/infra), OSS Python, OSS JavaScript, Studio, Pricing",
    "LangSmith deployment": "Setting up SSO, provisioning cloud resources, managing databases, helm/kubernetes/docker/AWS/GCP/Azure",
    "Other": "Sales inquiries, Spam, Unrelated issues"
    }
}
```

## Development

```bash
# Format code
uv run ruff format

# Check for issues
uv run ruff check

# Run tests
uv run pytest
```

