"""
Centralized prompt configs for conversation analysis
"""

SUMMARIZE_INSTR = """
Your job is to analyze this conversation and extract the key details about what the user is asking the AI assistant to do. 
Focus on capturing the main task, request, or purpose of the conversation in a clear, concise way.

Provide a structured summary in this format:
"[ACTION/TASK] with [SPECIFIC_TOPIC/SUBJECT] for [CONTEXT/PURPOSE]"

Examples:
- "help with writing Python code for data analysis project"
- "explain machine learning concepts for academic research"
- "create marketing content for social media campaign" 
- "debug software issues for web application development"
- "provide advice on career planning for recent graduate"
- "analyze financial data for investment decision making"
- "generate creative content for storytelling project"
- "answer questions about historical events for educational purposes"

Guidelines:
- Focus on what the user is asking the AI to do or help with
- Be specific about the subject matter or domain when clear
- Leave out redundant words like "User requested" or "I understand"
- Include context about the purpose, use case, or technical details when relevant to the domain
- Capture the core intent of the conversation
- Don't include any personally identifiable information (PII) like names, locations, phone numbers, email addresses
- Don't include any proper nouns
- Be clear, descriptive and specific 
- Keep it concise - aim for one clear sentence

Provide your summary of the support conversation in <answer> tags and select the most appropriate category for this conversation from the provided list:
{partitions}
"""

CRITERIA = """
The cluster name should be a specific, imperative sentence that helps teams, analysts, and stakeholders take immediate action.

**BUSINESS PURPOSE:** These clusters enable teams to:
- Route requests or issues to the right specialists quickly
- Prioritize which areas need immediate attention
- Identify patterns that require focus and resources
- Allocate effort based on frequency and importance

**NAMING REQUIREMENTS:**
Each cluster name must clearly indicate:
1. The main topic, domain, or subject area
2. The specific type of task or request
3. Technical or contextual details when relevant

**PRECISION REQUIREMENTS:**
- **Be maximally specific**: Use precise technical terms, specific technologies, exact problem types
- **Include technical details**: Programming languages, frameworks, specific tools, methodologies
- **Specify the exact task**: Not just "help" but "debug", "configure", "integrate", "optimize"
- **Add context when critical**: Target audience, use case, complexity level, domain area

**GOOD examples (actionable and specific):**
- "Debug React component state management issues in production applications"
- "Create SEO-optimized blog content for B2B software companies"
- "Explain advanced calculus integration techniques for engineering students"
- "Analyze customer churn patterns using machine learning for subscription businesses"
- "Generate Python data visualization scripts for financial time series analysis"

**BAD examples (too generic/vague):**
- "Fix technical problems" → What technology? What type of problem? What context?
- "Help with content" → What kind of content? What purpose? What format?
- "Answer questions" → About what topic? For what audience? At what level?
- "Debug code" → What language? What type of issue? What environment?

**TECHNICAL PRECISION GUIDELINES:**
- For programming: Include language, framework, specific error types, environment
- For writing: Include format, audience, purpose, style, domain
- For education: Include subject, level, specific concepts, target audience
- For business: Include industry, function, tools, specific objectives
- For analysis: Include data types, methods, tools, business context

Always specify enough detail that someone with the right expertise would immediately understand exactly what work is needed and what knowledge/tools to apply.
"""


NAME_CLUSTER_INSTR = """
You are tasked with summarizing a group of related requests into a short, precise, and accurate description and name. Your goal is to create a concise summary that captures the specific needs and distinguishes them from other types of requests.

Summarize all the requests into a clear, precise, two-sentence description in the past tense. Your summary should be specific to this group and distinguish it from the contrastive examples.

After creating the summary, generate a short name for the group of requests. This name should be at most ten words long and be specific about the topic, task type, or domain involved.

Focus on specificity rather than general terms. For instance, "Debug web application authentication errors", "Create social media marketing content", or "Explain advanced mathematics concepts" would be better than general terms like "Fix technical problems" or "Help with content". Be as specific as possible about the domain and task type while capturing the core need.

**CRITICAL CLUSTERING GUIDANCE:**
- Separate by TOPIC/DOMAIN first (programming vs writing vs education vs business)
- Then separate by TASK TYPE (debugging vs creating vs explaining vs analyzing)
- Then separate by CONTEXT (academic vs business vs personal vs creative)
- Then separate by COMPLEXITY (basic vs intermediate vs advanced)
- Finally separate by TECHNICAL DETAILS when relevant

Examples of good distinctions:
- "Programming debugging help" vs "Creative writing assistance" (different domains)
- "Basic concept explanations" vs "Advanced technical analysis" (different complexity)
- "Academic research help" vs "Business strategy advice" (different contexts)
- "Data analysis tasks" vs "Content creation requests" (different task types)

Present your output in the following format:
<summary> [Insert your two-sentence summary here] </summary>
<name> [Insert your generated short name here] </name>

The names you propose must follow these requirements:
<criteria>{criteria}</criteria>

Below are the related requests:
<answers>
{cluster_sample}
</answers>

For context, here are statements from nearby groups that are NOT part of the group you're summarizing:
<contrastive_answers>
{contrastive_sample}
</contrastive_answers>

Do not elaborate beyond what you say in the tags. Remember to focus on the specific topics, tasks, 
or domains that distinguish this group from others.
"""

PROPOSE_CLUSTERS_INSTR = """
You are tasked with creating higher-level categories based on a
given list of clusters and their descriptions. Your goal is to come up
with broader categories that could encompass one or more of the provided
clusters, while maintaining specificity and actionability.

First, review the list of clusters and their descriptions:
<cluster_list>
{cluster_list_text}
</cluster_list>

Your task is to create roughly {clusters_per_neighborhood} higher-level categories
that could potentially include one or more of the provided clusters. 

These categories should align with the LangChain support structure and be actionable for product teams:

These categories should be actionable and align with natural groupings:

**Main Category Types:**
1. **Topic/Domain Areas** - Programming, Writing, Education, Business, Science, etc.
2. **Task Types** - Creating, Debugging, Explaining, Analyzing, Planning, etc.
3. **Use Contexts** - Academic, Professional, Personal, Creative, Technical
4. **Complexity Levels** - Basic, Intermediate, Advanced
5. **Content Types** - Code, Text, Data, Visual, Audio, etc.

You can generate more or less than {clusters_per_neighborhood} names if appropriate. 
You should output at least {min_cpn} and at most {max_cpn}
names, with {clusters_per_neighborhood} as a target.

Guidelines for creating higher-level categories:
1. Analyze the topics, tasks, or contexts common to multiple clusters
2. Create names that clearly indicate what type of expertise or approach is needed
3. Ensure categories reflect natural groupings that would make sense organizationally
4. Use clear, descriptive language that immediately conveys the scope
5. Prioritize actionable distinctions over generic groupings

The names you propose must follow these requirements:
<criteria>{criteria}</criteria>

Before providing your final list, use a scratchpad to brainstorm and refine
your ideas. Think about the relationships between the given clusters and
potential overarching themes.

<scratchpad>
[Use this space to analyze the clusters and identify patterns. 
Consider how different clusters might be grouped together under broader categories. No longer than a paragraph or two.]
</scratchpad>

Now, provide your list of roughly {clusters_per_neighborhood} higher-level cluster names. Present your answer in 
the following format:
<answer>
1. [First higher-level cluster name]
2. [Second higher-level cluster name]
3. [Third higher-level cluster name]
...
{clusters_per_neighborhood}. [Last higher-level cluster name]
</answer>

Focus on creating meaningful, distinct, and actionable higher-level cluster names.
"""

DEDUPLICATE_CLUSTERS_INSTR = """
You are tasked with deduplicating a list of cluster names into distinct categories. 
Your goal is to create approximately {clusters_per_neighborhood} distinct categories that best represent the conversation patterns and are actionable.

Here are the inputs:
<cluster_names>
{cluster_text}
</cluster_names>

Number of distinct clusters to create: approximately {clusters_per_neighborhood}

IMPORTANT: Focus on actionable distinctions that help organize different types of work. For example, "programming debugging help", 
"creative writing assistance", "educational explanations", and "business analysis tasks" all 
require different expertise and approaches - keep them separate.

Follow these steps to complete the task:
1. Analyze the given list of cluster names to identify similarities,
patterns, and themes.
2. Group similar cluster names together based on their semantic meaning and
specific topic/task area, not just surface-level similarity.
3. For each group, select a representative name that best captures the
essence and tells people exactly what type of work this involves.
4. Only merge groups if they truly require the same expertise and approach. 
Maintain actionable distinctions for different types of work.
5. Prioritize keeping clusters separate if they involve different:
   - **Domains**: Programming vs Writing vs Education vs Business
   - **Task types**: Creating vs Debugging vs Explaining vs Analyzing
   - **Contexts**: Academic vs Professional vs Personal vs Creative
   - **Complexity**: Basic help vs Advanced problem-solving
   - **Content types**: Code vs Text vs Data vs Media
   - **Expertise needed**: Technical vs Creative vs Analytical vs Educational
6. Ensure that the final set of cluster names are distinct from each other
and collectively represent the diversity of the original list.
7. If you create new names for any clusters, make sure they clearly specify the
topic and task type so people know what expertise is needed.
8. You do not need to come up with exactly {clusters_per_neighborhood} names, but aim
for no less than {min_cpn} and no more than {max_cpn}. Within this range, output as many clusters as you
feel are necessary to accurately represent the variance in the original
list. Avoid outputting duplicate or near-duplicate clusters.
9. Prefer outputting specific, actionable cluster names over generic ones;
for example, "Debug web application programming issues" immediately tells people what type of expertise is needed.

**Examples of good actionable distinctions:**
- "Programming debugging and troubleshooting" vs "Creative writing and storytelling" (different domains)
- "Basic concept explanations" vs "Advanced technical analysis" (different complexity)
- "Academic research assistance" vs "Business strategy planning" (different contexts)
- "Data analysis and visualization" vs "Content creation and editing" (different expertise)

**What TO merge (same expertise/approach):**
- "Debug code errors" + "Fix programming issues" → "Debug programming and software issues"
- "Creative writing help" + "Story writing assistance" → "Creative writing and storytelling assistance"

**What NOT to merge (different expertise/approach):**
- "Programming tutorials" + "Creative writing help" (different domains)
- "Basic math explanations" + "Advanced data analysis" (different complexity)

The names you propose must follow these requirements:
<criteria>{criteria}</criteria>

Before providing your final answer, use the <scratchpad> tags to think
through your process, specifically identifying the different domains and task types that would require different expertise.

Present your final answer in the following format: 
<answer>
1. [First actionable cluster name]
2. [Second actionable cluster name]
3. [Third actionable cluster name]
...
N. [Nth actionable cluster name]
</answer>

Remember, your goal is to create approximately {target_clusters} actionable cluster names that help people immediately understand what type of work and expertise is involved.
"""

ASSIGN_CLUSTER_INSTR = """
You are tasked with categorizing a specific cluster into one of the provided higher-level categories. 
Focus on matching the type of work and expertise needed. Your goal is to determine which higher-level cluster best
fits the given specific cluster based on its name and description.

First, carefully review the following list of higher-level clusters:
<higher_level_clusters>
{higher_level_text}
</higher_level_clusters>

To categorize the cluster:
1. **Identify the main topic/domain** (Programming, Writing, Education, Business, etc.)
2. **Determine the task type** (Creating, Debugging, Explaining, Analyzing, etc.)
3. **Note the context** (Academic, Professional, Personal, Creative, etc.)
4. **Consider the complexity level** (Basic, Intermediate, Advanced)
5. **Match to the category that involves the same type of work and expertise**

Be precise with categorization. For example:
- Python web application debugging issues → Programming debugging and troubleshooting assistance
- Creative fiction writing and character development → Creative writing and storytelling assistance  
- Advanced calculus concept explanations for university students → Educational tutoring and concept explanation
- Financial data analysis for quarterly business planning → Business data analysis and strategic planning

First, use the <scratchpad> tags to think through your reasoning:

<scratchpad>
Think step by step: What domain is this about? What type of task? What level of expertise is needed? Which higher-level category best matches these requirements?
</scratchpad>

Then, provide your answer in the following format:
<answer>
[Full name of the chosen cluster, exactly as listed in the higher-level
clusters above, without enclosing <cluster> tags]
</answer>
"""

RENAME_CLUSTER_INSTR = """
You are tasked with summarizing a group of related cluster names into
a short, precise, and accurate overall description and name.

Summarize the work into a clear, precise, two-sentence description in the past tense. 
Your summary should specify what type of work this involves and what expertise is typically needed.

After creating the summary, create a short name that immediately tells people:
- **What domain** this covers (Programming, Writing, Education, Business, etc.)
- **What type of task** this involves (Creating, Debugging, Explaining, Analyzing, etc.)
- **What context** this is for (Academic, Professional, Personal, Creative, etc.)

Focus on actionable specificity. For instance:
- "Debug programming and software development issues" → Programming expertise needed
- "Create marketing and promotional content" → Writing/Marketing expertise needed  
- "Explain educational concepts and provide tutoring" → Teaching/Educational expertise needed
- "Analyze business data and provide strategic insights" → Business/Analytics expertise needed

Consider typical expertise areas:
- **Programming/Technical**: Coding, debugging, software development, system administration
- **Writing/Content**: Creative writing, marketing content, documentation, editing
- **Education/Tutoring**: Concept explanation, academic help, skill teaching
- **Business/Strategy**: Analysis, planning, decision making, process optimization
- **Creative/Design**: Visual content, artistic projects, creative problem solving
- **Research/Analysis**: Data analysis, research assistance, information synthesis

The goal is helping people immediately understand: "This involves this type of work, requiring this kind of expertise."

Present your output in the following format:
<summary> [Insert your two-sentence summary that specifies work type and expertise needed] </summary>
<name> [Insert your actionable name here, with no period or trailing punctuation] </name>

The name you choose must follow these requirements:
<criteria>{criteria}</criteria>

Below are the related statements:
<answers>
{cluster_list_text}
</answers>

Do not elaborate beyond what you say in the tags. Ensure your summary and name help people immediately understand the type of work and expertise involved.
"""

# changed {clusters_per_neighborhood} to {target_clusters}, twice

# Default partitions for evaluations

# Evals
CATEGORY_RELEVANCE = """
You are evaluating whether conversation fits its assigned category.
CONVERSATION SUMMARY: {{summary}}
ASSIGNED CATEGORY: {{category}}
PARTITIONS:
{partitions}
TASK: Does this conversation summary fit well within the assigned category?
Consider:
- Does the conversation topic align with the category's scope?
- Would a support team for this category be the right team to handle this request?
- Is this conversation clearly about the assigned product area?
Rate this as either:
- CORRECT (1): The conversation clearly fits the assigned category
- INCORRECT (0): The conversation does not fit the assigned category
Provide your rating as either 1 or 0.
"""


HIERARCHICAL_FIT = """
You are evaluating whether a conversation fits under its assigned base cluster.
BASE CLUSTER: {current_cluster} 
CONVERSATION SUMMARY: {summary} 
TASK: Does this conversation logically belong under the assigned base cluster?
Consider:
- Does the conversation topic align with the cluster's scope?
- Would a support team for this cluster be the right team to handle this request?
- Is this a sensible organizational grouping?
Rate this as either:
- CORRECT (1): The conversation clearly belongs under the base cluster
- INCORRECT (0): The conversation does not belong under the base cluster
Provide your rating as either 1 or 0.
"""

BEST_FIT = """
    You are evaluating whether a conversation is assigned to the best possible base cluster.

CONVERSATION SUMMARY: {summary}
CURRENT ASSIGNMENT: {current_cluster}

ALL AVAILABLE BASE CLUSTERS:
{all_base_clusters_text}

TASK: Is the current assignment the BEST fit among all available options?

Consider:
- Which cluster would most naturally handle this type of request?
- Does the current assignment make the most sense organizationally?
- Would any other cluster be a better fit for this conversation?

Rate this as either:
- OPTIMAL (1): The current assignment is the best fit among all options
- SUBOPTIMAL (0): Another cluster would be a better fit for this conversation

If rating as SUBOPTIMAL, the conversation would fit better in a different cluster.

Provide your rating as either 1 or 0. If you provide a rating of 0, provide the name of the cluster where it would fit better.
"""

EXCLUSIVE_FIT = """
Analyze this conversation and identify ALL clusters it could reasonably belong to.

CONVERSATION SUMMARY: {summary}
CURRENT ASSIGNMENT: {current_cluster}

ALL AVAILABLE CLUSTERS: 
{all_base_clusters_text}

TASK: List ALL clusters (including the current one) where this conversation could reasonably fit. 
A conversation "fits" if that team could appropriately handle the request.

Rate each conversation as either:
- EXCLUSIVE FIT(1): This conversation clearly belongs to its current cluster and would not fit well in other clusters.
- AMBIGUOUS FIT (0): This conversation could reasonably belong to multiple clusters or its cluster assignment is unclear.
"""

DEDUPLICATE = """
Here's a list of {total_base_clusters} topics extracted from a series of conversations:
{all_base_clusters_text}

TASK: Count how many truly unique topics there are after consolidating semantically equivalent ones.

Two topics should be merged if they handle very similar requests or represent the same core problem area.

Think through this step by step:
1. Group together topics that are essentially the same
2. Count the number of unique groups  
3. Calculate: unique_topics / {total_base_clusters}

At the end, provide the final calculation in this format:
FINAL SCORE: [decimal number]

For example: FINAL SCORE: 0.73
"""

"""
{
  "summary": "debugging help with LangSmith SDK tracing for Python implementation",
  "category": "LangSmith product",
  "clustering": {
    "level_0": {
      "id": 5,
      "name": "Debug LangSmith Python SDK tracing integration errors"
    },
    "level_1": {
      "id": 2,
      "name": "Handle LangSmith SDK Integration and Tracing Issues"
    }
  }
}
"""
