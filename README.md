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
Choose a dataset or tracing project to run Clio clustering on and create a `config.json` file, or skip this step and test with `configs/tracing_project_example.json`.
```json
{
  "project_name": "chat-langchain-v3",
  "start_time": "2025-07-19T00:00:00Z",
  "end_time": "2025-07-20T00:00:00Z",
  "hierarchy": [16, 5],
  "summary_prompt": "Summarize this run: {{run.inputs}} {{run.outputs}}", 
  "save_path": "./tracing_project_results",
  "sample": 100,
  "partitions": null
}
```

**Writing your custom config:**
Choose a data source:
- `dataset_name`: LangSmith dataset name  
- `project_name`: LangSmith tracing project name

If you specified a `project_name`, you may also include time filtering options (only one method allowed):
- `start_time` and `end_time`: explicit ISO format timestamps (e.g., "2025-07-19T00:00:00Z")
- `hours`: integer specifying "last X hours" (e.g., `24` for last 24 hours)
- `days`: integer specifying "last X days" (e.g., `7` for last 7 days)

If no time filtering is specified, defaults to last 1 hour.

You must also specify:
- `hierarchy`: How many clusters to create at each level and how many levels total [base_level, middle_level, top_level] (3 levels in this example)
- `summary_prompt`: Pass in the example or run to summarize (for example {{run}} or {{example.inputs}}) and instruct an LLM on how to summarize it.
- `save_path`: Where to save the results
- `sample`: the maximum number of examples or runs to return

Additionally, you can include:
- `partitions`: overarching areas to sort your examples or runs into


### 3. Generate Clio clusters
```bash
uv run open-clio generate --config path/to/config
```

This will:
- Load examples from your dataset or threads from your tracing project
- Summarize each item
- Create clusters at multiple levels
- Launch a web interface to explore the results

### 4. Explore your results
When clustering is complete, the command above automatically opens a web browser where you can navigate through your clusters, from high-level categories down to individual examples.

You can also explore the output files in `save_path` for a quick overview of clusters:
- `combined.csv`: All conversations with their cluster assignments
- `level_{x}_clusters.csv`: Information about clusters created at each level of the hierarchy (lowest level is most granular, highest level is most broad)


## Example Walkthrough

This example is pre-configured for a dataset of customer support conversations:

1. **Run**: `open-clio generate --config configs/customer_support.json` 
2. **Explore**: The web interface shows clusters like:
   - "Collect financial invoices and receipts from business organizations for accounting record management"
   - "Manage enterprise SaaS platform operations"
   - "Establish security frameworks and vulnerability management systems for open-source AI development projects"

   
## How to:

### Write a summary prompt

Your summary prompt tells the LLM how to extract key information from each example or run, especially any domain or application specific info that the example's summary must retain. More specific summary prompts create more accurate and precise clusters.

Use a mustache template to include the fields of your example or run to summarize, for example ```{{run.inputs.messages}}``` or ```{{example.inputs.0}}```

This an example for a chatbot:

```json
"summary_prompt": "Your job is to analyze this conversation {{example.inputs}}\nExtract the key details about what the user is asking the AI assistant to do.\nFocus on capturing the main task, request, or purpose of the conversation in a clear, concise way.\n\nProvide a structured summary in this format:\n'[ACTION/TASK] with [SPECIFIC_TOPIC/SUBJECT] for [CONTEXT/PURPOSE]'\n\nGuidelines:\n- Leave out redundant words like 'User requested' or 'I understand'\n- Include context about the purpose, use case, or technical details when relevant to the domain\n- Keep it concise - aim for one clear sentence"
```

### Set a hierarchy

Your hierarchy defines how many levels to create and how many clusters to create at each level. Use a list of decreasing numbers:

```json
"hierarchy": [15, 8, 3]
```

**Structure:** `[base_clusters, middle_clusters, top_clusters]`

**Examples:**
- `[15, 8, 3]` - 15 base clusters → 8 mid-level clusters → 3 top level clusters
- `[25, 5]` - 25 base clusters → 5 top level clusters 
- `[10]` - 10 base clusters only

**Guidelines:**
- Start with 1-3 levels for meaningful results, or 1-2 levels if you are using partitions.
- Base level should be ≤ √(number of examples)
- Each level should have fewer clusters than the previous
- Match top-level count to your partition count (if using partitions)

**For 100 examples:** Try `[16, 5]`

### Evaluate clusters

```bash
open-clio evaluate --config path/to/config.json
```
If you generated clusters for a dataser, you can use this command to evaluate clustering quality:

**Partition Relevance** - Checks if conversations are assigned to the correct partition (when using partitions)

**Hierarchical Fit** - Tests if conversations belong in their assigned clusters at each level

**Best Fit** - Verifies if conversations are assigned to the optimal cluster among all available options

**Exclusive Fit** - Measures whether conversations fit exclusively in their assigned cluster or could belong to multiple clusters

**Uniqueness Score** - Evaluates how distinct your clusters are (higher = more unique clusters)

### Define partitions upfront

Partitions pre-categorize your data into broad domains before clustering. Add a `partitions` object to your `config.json`:

```json
{
    "partitions": {
        "Technical Issues": "Software bugs, system errors, and technical troubleshooting",
        "Account & Billing": "Payment issues, subscription management, and billing questions",
        "Feature Requests": "New feature suggestions and enhancement requests",
        "General Support": "Product information and general inquiries"
    }
}
```

**Structure:** Key = partition name, Value = description of what belongs there.

**Best practices:**
- Cover all major categories in your domain
- Be specific in descriptions to help the LLM categorize correctly
- Match partition count to your top-level cluster count
- Use clear, actionable names

**When to use:** You have domain knowledge and want to ensure related examples are grouped together.

**When not to use:** You're exploring unknown patterns or want completely data-driven clustering.

If no partitions are defined, all examples go into a single "Default" partition.


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

