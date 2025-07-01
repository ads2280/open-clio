"""
Centralized prompt configs for customer service conversation analysis
"""
# should probably add these to prompt hub

PRIVACY_INSTR = """
When answering, do not include any personally identifiable information (PII), like
   names, locations, phone numbers, email addressess, and so on. When answering,
   do not include any proper nouns. Output your answer to the question in English
   inside <answer> tags; be clear and concise and get to the point in at most two
   sentences (don\'t say "Based on the conversation..." and avoid mentioning
   the AI system/chatbot). For example:
   <examples>
   The user asked for help with LangSmith tracing integration issues.
   The user asked for assistance with LangGraph platform deployment configuration. It took several
   attempts to resolve the authentication errors.
   The user requested billing information about subscription upgrades and the support agent provided pricing details.
   </examples>

"""


SUMMARIZE_INSTR = """Your job is to analyze this support conversation and extract the key details. Focus on: 

1. **Product**: Which specific product/component is this about?
   - LangChain OSS: langchain-{x} open-source packages like 'langchain', 'langchain-openai', 'langchain-community', etc., across both Python and JS/TS. \
   The LangChain open source packages are a colleciton of Python/JS libraries for working with LLMs. \
   They integrates with various LLMs, databases and APIs. Note to distinguish these from LangGraph OSS questions. \
   LangChain OSS questions will often have to do with integrations, while LangGraph questions have to do with LLM agent orchestration.
   - LangGraph OSS: The 'langgraph' Python and JS/TS open source packages. \
   LangGraph is a framework for building LLM agents. \
   Note to distinguish LangGraph OSS from LangGraph Platform, a commercial platform for deploying LLM agents built on top of the open source packages.
   - LangGraph Platform/Studio: LangGraph Platform is a commercial product for deploying LLM agents built with LangGraph OSS. \
   LangGraph Studio is a visual debugger for any LLM agents built with LangGraph OSS. \
   Any conversations related to the features of these products or deployment of these products should go under this category.
   - LangSmith product: LangSmith product features. LangSmith is an observability and evals platform for AI agents. \
   It integrates with LangChain and LangGraph but is a standalone product. Any questions related to tracing, evals, the LangSmith SDK, the LangSmith UI, playground, datasets, and prompt hub should go here. Note to distinguish this from LangSmith deployment and Admin/Account management, which have to do with setting up the infrastructure to deploy the LangSmith service or administrative tasks within LangSmith and not the actual product features.
   - LangSmith deployment: Any issues related to deploying LangSmith the platform (not instrumenting an application to trace to LangSmith, which should go under LangSmith product). \
   These might have to do with setting up SSO, provisioning cloud resources, managing databases, helm/kubernetes/docker/AWS/GCP/Azure/Clickhouse/Postgres/Redis/S3.
   - Admin/Account management: Any issues related to billing, sign up, inviting members, usage quotas, receipts, account management, etc.
   - Unrelated: Any issues unrelated to any of the products.
   
2. **Issue Type**: What kind of problem or request is this?
   - Setup/Installation
   - Integration (connecting with other systems)
   - Configuration (settings, parameters)
   - Debugging/Troubleshooting
   - Documentation Gap (missing or unclear docs)
   - Feature Request
   - Performance/Optimization
   - Best Practices/Usage Questions
   - API Usage
   - Deployment issues
   - Authentication/Access Problems
   - Billing/Account Issues
   - Data Management
   - Version/Compatibility Issues

3. **Language/Framework** (if applicable): Python, JavaScript, specific frameworks

Provide a structured summary in this format:
"User requested [ISSUE_TYPE] help with [SPECIFIC_PRODUCT] for [LANGUAGE/FRAMEWORK] implementation"

Examples:
- "User requested debugging help with LangSmith SDK tracing for intermediate Python implementation"
- "User requested setup help with LangGraph Platform for basic deployment configuration"
- "User requested integration help with LangChain Python Library for advanced RAG implementation"

Guidelines:
- Be specific about the LangChain product/component
- Include the programming language when relevant
- Capture the technical nature of the request
- Don't include any personally identifiable information (PII) like names, locations, phone numbers, email addresses
- Don't include any proper nouns
- Be clear, descriptive and specific 

Provide your summary of the support conversation in <answer> tags. Provide only the answer with no other commentary.
"""

CRITERIA = """
The cluster name should be a specific, imperative sentence that helps customer support teams, product managers, and documentation teams take immediate action.

**BUSINESS PURPOSE:** These clusters enable teams to:
- Route support tickets to the right product specialists quickly
- Prioritize which documentation gaps to fill first  
- Identify product areas needing immediate engineering attention
- Allocate resources based on user pain frequency

**NAMING REQUIREMENTS:**
Each cluster name must clearly indicate:
1. Which LangChain product/component (LangSmith SDK, LangGraph Platform, LangChain Python, etc.)
2. The specific issue type (integration, debugging, setup, configuration, deployment, etc.)
3. Technical context when relevant (Python, JavaScript, API, tracing, evaluation, etc.)

**GOOD examples (actionable for teams):**
- "Debug LangSmith Python SDK tracing integration errors" → LangSmith team, SDK expertise needed
- "Configure LangGraph Platform deployment for production scaling" → LangGraph Platform team, DevOps docs needed
- "Resolve LangChain JavaScript memory management issues" → LangChain JS team, performance optimization needed
- "Setup LangSmith evaluation workflows for custom datasets" → LangSmith Evaluation team, workflow documentation needed
- "Handle Admin billing subscription and payment issues" → Admin team, billing system fixes needed

**BAD examples (not actionable):**
- "Fix technical problems" → Which team? Which product? What type of fix?
- "Help with platform issues" → Which platform? What specific problem?
- "Debug application problems" → Which application? What component?

Always specify the exact LangChain product and technical area so teams know who should handle it and what type of solution is needed."""


NAME_CLUSTER_INSTR = """
You are tasked with summarizing a group of related LangChain/LangSmith support requests into a short, precise, and accurate description and name. Your goal is to create a concise summary that captures the specific technical support needs and distinguishes them from other types of support requests.

Summarize all the requests into a clear, precise, two-sentence description in the past tense. Your summary should be specific to the LangChain ecosystem (LangChain OSS, LangSmith, LangGraph) and distinguish it from the contrastive examples of other support clusters.

After creating the summary, generate a short name for the group of requests. This name should be at most ten words long and be specific about the LangChain component, feature, or technical issue involved.

Focus on LangChain ecosystem specificity rather than general terms. For instance, "Debug LangSmith evaluation SDK integration errors", "Configure LangGraph platform deployment settings", or "Handle LangChain Python memory management issues" would be better than general terms like "Fix technical problems" or "Help with platform issues". Be as specific as possible about the LangChain component and technical area while capturing the core support need.

Categories to consider:
- Admin: Billing, Authentication, Account Management, Data Deletion, Security/Privacy
- LangChain OSS: Python library, JavaScript library, LangMem and other components
- LangSmith: Evaluation, Dashboards, Annotations, Datasets & Experiments, Playground, SDK/Tracing, Prompt Hub, Automation Rules, Observability, Pricing, Administration
- LangGraph: Platform (deployment/infra), OSS Python, OSS JavaScript, Studio, Pricing
- Other: Sales inquiries, Spam

**CRITICAL CLUSTERING GUIDANCE:**
- Separate issues by PRODUCT first (LangChain vs LangSmith vs LangGraph vs Admin)
- Then separate by LANGUAGE (Python vs JavaScript) when applicable
- Then separate by FEATURE AREA (SDK/Tracing vs Evaluation vs Platform vs Studio)
- Then separate by ISSUE TYPE (setup vs debugging vs configuration vs deployment)
- Finally separate by COMPLEXITY (basic usage vs advanced integration)

Examples of good product/feature distinctions:
- "LangSmith Python SDK integration" vs "LangSmith JavaScript SDK integration" (different languages)
- "LangSmith evaluation setup" vs "LangSmith tracing configuration" (different features)
- "LangGraph Platform deployment" vs "LangGraph Python library usage" (different components)
- "Admin billing issues" vs "Admin authentication problems" (different admin areas)

Present your output in the following format:
<summary> [Insert your two-sentence summary here] </summary>
<name> [Insert your generated short name here] </name>

The names you propose must follow these requirements:
<criteria>{criteria}</criteria>

Below are the related LangChain/LangSmith support requests:
<answers>
{cluster_sample}
</answers>

For context, here are statements from nearby support groups that are NOT part of the group you're summarizing:
<contrastive_answers>
{contrastive_sample}
</contrastive_answers>

Do not elaborate beyond what you say in the tags. Remember to focus on the specific LangChain components, features, 
or technical issues that distinguish this group from others in the LangChain ecosystem.

"""

PROPOSE_CLUSTERS_INSTR = """
You are tasked with creating higher-level LangChain support categories based on a
given list of LangChain/LangSmith support clusters and their descriptions. Your goal is to come up
with broader support categories that could encompass one or more of the provided
clusters, while maintaining LangChain ecosystem specificity and actionability for support teams.

First, review the list of LangChain support clusters and their descriptions:
<cluster_list>
{cluster_list_text}
</cluster_list>

Your task is to create roughly {clusters_per_neighborhood} higher-level LangChain support categories
that could potentially include one or more of the provided clusters. 

These categories should align with the LangChain support structure and be actionable for product teams:

**Main Categories:**
1. **Admin** - Billing & Refunds, Authentication & Access, General Account Management, Data Deletion, Security/Privacy/Compliance
2. **LangChain** - OSS Python, OSS JavaScript, Other OSS components (LangMem, etc.)
3. **LangSmith** - Evaluation, Dashboards, Annotations, Datasets & Experiments, Playground, SDK/Tracing, Prompt Hub, Automation Rules, Observability, Pricing, Administration
4. **LangGraph** - Platform (deployment/infra), OSS Python, OSS JavaScript, Studio, Pricing
5. **Other** - Sales, Spam

**Subcategory Examples (actionable for teams):**
- "LangSmith SDK Integration and Tracing Issues" → LangSmith SDK team
- "LangGraph Platform Deployment and Infrastructure" → LangGraph Platform team
- "LangChain Python Library Usage and Implementation" → LangChain Python team
- "Admin Billing and Account Management" → Admin/Finance team
- "LangSmith Evaluation and Dataset Management" → LangSmith Evaluation team

**Keep these distinctions separate (different teams/solutions needed):**
- Different products (LangChain vs LangSmith vs LangGraph vs Admin)
- Different languages (Python vs JavaScript)
- Different features (Tracing vs Evaluation vs Deployment vs Studio)
- Different issue types (Setup vs Integration vs Debugging vs Documentation)
- Different components (SDK vs Platform vs Library vs Dashboard)

You can generate more or less than {clusters_per_neighborhood} names if appropriate. 
You should output at least {min_cpn} and at most {max_cpn}
names, with {clusters_per_neighborhood} as a target.

Guidelines for creating higher-level support categories:
1. Analyze the LangChain products, features, or support areas common to multiple clusters
2. Create names specific to LangChain ecosystem that tell teams exactly what expertise is needed
3. Ensure categories align with the main LangChain support structure (Admin/LangChain/LangSmith/LangGraph/Other)
4. Use clear, product-specific language that support teams can immediately act on
5. Prioritize actionable distinctions over generic groupings

The names you propose must follow these requirements:
<criteria>{criteria}</criteria>

Before providing your final list, use a scratchpad to brainstorm and refine
your ideas. Think about the relationships between the given clusters and
potential overarching LangChain support themes.

<scratchpad>
[Use this space to analyze the clusters, analyze the LangChain products, features, and support patterns. 
Consider how different clusters might be grouped together under broader LangChain support categories that would be actionable for different product teams. No longer than a paragraph or two.]
</scratchpad>

Now, provide your list of roughly {clusters_per_neighborhood} higher-level cluster names. Present your answer in 
the following format:
<answer>
1. [First higher-level LangChain support cluster name]
2. [Second higher-level LangChain support cluster name]
3. [Third higher-level LangChain support cluster name]
...
{clusters_per_neighborhood}. [Last higher-level cluster name]
</answer>

Focus on creating meaningful, distinct, and actionable higher-level cluster names that align with 
the LangChain support structure and help teams prioritize their work.
"""

DEDUPLICATE_CLUSTERS_INSTR = """
You are tasked with deduplicating a list of LangChain support cluster names into distinct support categories. 
Your goal is to create approximately {clusters_per_neighborhood} distinct support categories that best represent the LangChain ecosystem support patterns and are actionable for product teams.

Here are the inputs:
<cluster_names>
{cluster_text}
</cluster_names>

Number of distinct clusters to create: approximately {clusters_per_neighborhood}

IMPORTANT: Focus on actionable distinctions that help different teams prioritize their work. For example, "LangSmith evaluation issues", 
"LangGraph deployment problems", "LangChain Python library bugs", and "billing account management" are all 
LangChain support but require different teams and solutions - keep them separate.

Follow these steps to complete the task:
1. Analyze the given list of cluster names to identify similarities,
patterns, and themes WITHIN the LangChain support domain.
2. Group similar cluster names together based on their semantic meaning and
specific LangChain product/feature, not just because they're all "LangChain support".
3. For each group, select a representative name that best captures the
specific LangChain support essence and tells teams exactly what expertise is needed.
4. Only merge groups if they truly require the same team and solution type. 
Maintain actionable distinctions for different teams.
5. Prioritize keeping clusters separate if they involve different:
   - **Teams**: LangChain vs LangSmith vs LangGraph vs Admin teams
   - **Products**: Different LangChain ecosystem products
   - **Languages**: Python vs JavaScript (different developer expertise)
   - **Feature areas**: Evaluation vs Deployment vs Tracing vs Billing
   - **Solution types**: Documentation vs Product fixes vs API changes
   - **Issue complexity**: Basic setup vs Advanced integration
6. Ensure that the final set of cluster names are distinct from each other
and collectively represent the LangChain support diversity of the original list.
7. If you create new names for any clusters, make sure they specify the
LangChain product/feature and support type clearly so teams know who should handle it.
8. You do not need to come up with exactly {clusters_per_neighborhood} names, but aim
for no less than {min_cpn} and no more than {max_cpn}. Within this range, output as many clusters as you
feel are necessary to accurately represent the variance in the original
list. Avoid outputting duplicate or near-duplicate clusters.
9. Prefer outputting specific, actionable LangChain support cluster names over generic ones;
for example, "Debug LangSmith SDK tracing integration" immediately tells the LangSmith SDK team what to work on.

**Examples of good actionable distinctions:**
- "LangSmith evaluation configuration and setup" vs "LangSmith dashboard UI and functionality issues" (different LangSmith teams)
- "LangGraph Python library implementation" vs "LangGraph platform deployment and infrastructure" (different expertise needed)
- "Admin billing and subscription management" vs "Admin authentication and access control" (different admin solutions)
- "LangChain Python performance optimization" vs "LangChain JavaScript integration problems" (different language expertise)

**What TO merge (same team/solution):**
- "Debug LangSmith tracing" + "Fix LangSmith tracing errors" → "Debug LangSmith SDK tracing issues"
- "LangGraph deployment help" + "LangGraph platform setup" → "LangGraph Platform deployment and configuration"

**What NOT to merge (different teams/solutions):**
- "LangSmith Python SDK" + "LangSmith JavaScript SDK" (different language teams)
- "LangGraph Platform deployment" + "LangGraph Studio problems" (different product components)

The names you propose must follow these requirements:
<criteria>{criteria}</criteria>

Before providing your final answer, use the <scratchpad> tags to think
through your process, specifically identifying the different LangChain
products and support types in the list that would require different teams or solutions.

Present your final answer in the following format: 
<answer>
1. [First actionable LangChain support cluster name]
2. [Second actionable LangChain support cluster name]
3. [Third actionable LangChain support cluster name]
...
N. [Nth actionable LangChain support cluster name]
</answer>

Remember, your goal is to create approximately {target_clusters} actionable cluster names that help teams immediately understand what work needs to be prioritized and who should handle it.
"""

ASSIGN_CLUSTER_INSTR = """
You are tasked with categorizing a specific LangChain support cluster into one of the provided higher-level LangChain support categories. 
Focus on actionable team assignment - which team should handle this and what expertise do they need? Your goal is to determine which higher-level cluster best
fits the given specific cluster based on its name and description.

First, carefully review the following list of higher-level clusters:
<higher_level_clusters>
{higher_level_text}
</higher_level_clusters>

To categorize the LangChain support cluster for actionable team assignment:
1. **Identify the main LangChain product** (LangChain OSS, LangSmith, LangGraph, Admin, Other)
2. **Determine the specific component** (SDK, Evaluation, Platform, Billing, Studio, etc.)
3. **Note the technical expertise needed** (Python, JavaScript, DevOps, API integration, etc.)
4. **Consider the solution type** (Documentation, Product fix, API change, Configuration help)
5. **Match to the category that requires the same team and expertise**

Be precise with team assignment. For example:
- LangSmith tracing issues → LangSmith SDK team (not general debugging team)
- LangGraph deployment problems → LangGraph Platform team (not general infrastructure team)  
- Billing disputes → Admin/Finance team (not general account team)
- Python library bugs → LangChain Python team (not JavaScript team)

First, use the <scratchpad> tags to think through your reasoning:

<scratchpad>
Think step by step: What LangChain product is this about? What specific component/feature? What type of expertise is needed? Which team should handle this? Which higher-level category best matches these requirements?
</scratchpad>

Then, provide your answer in the following format:
<answer>
[Full name of the chosen cluster, exactly as listed in the higher-level
clusters above, without enclosing <cluster> tags]
</answer>
"""

RENAME_CLUSTER_INSTR = """
You are tasked with summarizing a group of related LangChain support cluster names into
a short, precise, and accurate overall description and name that is actionable for product teams.

Summarize the LangChain support work into a clear, precise, two-sentence description in the past tense. 
Your summary should specify which teams need to be involved and what type of solutions are typically needed.

After creating the summary, create a short name that immediately tells teams:
- **Which product team** should handle this (LangSmith, LangGraph, LangChain, Admin)
- **What specific area** needs attention (SDK, Platform, Evaluation, Billing, etc.)
- **What type of work** is needed (Integration, Debugging, Configuration, Documentation, etc.)

Focus on actionable specificity. For instance:
- "Debug LangSmith SDK tracing integration issues" → LangSmith SDK team, integration expertise needed
- "Configure LangGraph Platform deployment and scaling" → LangGraph Platform team, DevOps/deployment expertise needed  
- "Handle Admin billing subscription and payment issues" → Admin/Finance team, billing system expertise needed
- "Resolve LangChain Python library performance problems" → LangChain Python team, performance optimization needed

Consider the LangChain support structure for team assignment:
- **Admin Team**: Billing & Refunds, Authentication & Access, Account Management, Data Deletion, Security/Privacy
- **LangChain Teams**: Python library team, JavaScript library team, LangMem team
- **LangSmith Teams**: Evaluation team, Dashboards team, Annotations team, SDK/Tracing team, Prompt Hub team, etc.
- **LangGraph Teams**: Platform team (deployment/infra), Python library team, JavaScript library team, Studio team
- **Other Teams**: Sales, Spam/Abuse

The goal is helping teams immediately understand: "This cluster is for us, we need this type of expertise, and we should prioritize this type of solution."

Present your output in the following format:
<summary> [Insert your two-sentence summary that specifies teams and solution types needed] </summary>
<name> [Insert your actionable team-focused name here, with no period or trailing punctuation] </name>

The name you choose must follow these requirements:
<criteria>{criteria}</criteria>

Below are the related LangChain support statements:
<answers>
{cluster_list_text}
</answers>

Do not elaborate beyond what you say in the tags. Ensure your summary and name help teams immediately understand their responsibilities and priorities.
"""

# changed {clusters_per_neighborhood} to {target_clusters}, twice
